# Laravel integration

This is the use case the project was built for: seeding a Laravel app with
believable user avatars, and serving a stable avatar per user afterwards.

Everything below is plain Laravel 11/12 with the `Http` facade. No package to
install.

## Configuration

`config/ppg.php`:

```php
<?php

return [
    // No trailing slash. On a shared Docker network this is http://api:8000.
    'url' => rtrim(env('PPG_URL', 'http://127.0.0.1:8000'), '/'),

    // Matches PPG_API_KEY on the generator. Null means the service is open,
    // which is only acceptable on a private network.
    'key' => env('PPG_KEY'),

    // Generation takes about 8 seconds on a 12GB card and minutes on CPU.
    'timeout' => (int) env('PPG_TIMEOUT', 120),

    'size' => (int) env('PPG_AVATAR_SIZE', 256),
    'format' => env('PPG_AVATAR_FORMAT', 'webp'),
];
```

`.env`:

```dotenv
PPG_URL=http://127.0.0.1:8000
PPG_KEY=
PPG_AVATAR_SIZE=256
```

## The client

`app/Services/AvatarClient.php`:

```php
<?php

namespace App\Services;

use Illuminate\Http\Client\ConnectionException;
use Illuminate\Http\Client\PendingRequest;
use Illuminate\Http\Client\RequestException;
use Illuminate\Http\Client\Response;
use Illuminate\Support\Facades\Http;
use RuntimeException;

class AvatarClient
{
    /**
     * Generate one avatar and return the full result, including the persona.
     *
     * @param  array<string, mixed>  $attributes  Any subset of /v1/options axes.
     * @return array<string, mixed>
     */
    public function generate(array $attributes = [], int|string|null $seed = null): array
    {
        $payload = $attributes;

        if ($seed !== null) {
            $payload['seed'] = $seed;
        }

        $response = $this->http()
            ->post('/v1/avatars?wait=' . config('ppg.timeout'), $payload ?: (object) []);

        // 200 = finished. 202 = the wait expired and we were handed a job.
        if ($response->status() === 202) {
            return $this->awaitJob($response->json('id'));
        }

        return $this->ok($response)->json();
    }

    /**
     * Queue a diverse batch. Returns the job payload; poll it with job().
     *
     * @return array<string, mixed>
     */
    public function batch(int $n, int|string|null $seed = null, array $overrides = []): array
    {
        return $this->ok($this->http()->post('/v1/avatars/batch', [
            'n' => $n,
            'diversity' => 'even',
            'seed' => $seed,
            'overrides' => (object) $overrides,
        ]))->json();
    }

    /** @return array<string, mixed> */
    public function job(string $id): array
    {
        return $this->ok($this->http()->get("/v1/jobs/{$id}"))->json();
    }

    /**
     * The deterministic URL for a key. Nothing is generated until it is fetched.
     */
    public function seedUrl(string $key, ?int $size = null): string
    {
        return sprintf(
            '%s/v1/avatars/by-seed/%s?size=%d&format=%s',
            config('ppg.url'),
            rawurlencode($key),
            $size ?? config('ppg.size'),
            config('ppg.format'),
        );
    }

    /**
     * The stable key for a user.
     *
     * A keyed hash, not a bare md5 of the address. The key travels in an image
     * URL, through access logs and into browser history, and an unsalted hash
     * of a common email address is reversible in seconds - it is the address,
     * with extra steps. HMAC with the application key keeps it opaque to
     * anyone who sees the URL while staying stable for the user.
     *
     * Lowercased and trimmed first so casing never forks the avatar. If you
     * ever rotate APP_KEY, every avatar changes; a random per-user column is
     * the alternative if that matters.
     */
    public function keyFor(string $email): string
    {
        return hash_hmac('sha256', mb_strtolower(trim($email)), config('app.key'));
    }

    /** Raw image bytes, for storing a copy locally or proxying. */
    public function fetch(string $key, ?int $size = null): string
    {
        return $this->ok(
            $this->http()->get('/v1/avatars/by-seed/' . rawurlencode($key), [
                'size' => $size ?? config('ppg.size'),
                'format' => config('ppg.format'),
            ])
        )->body();
    }

    private function http(): PendingRequest
    {
        return Http::baseUrl(config('ppg.url'))
            ->connectTimeout(5)
            ->timeout(config('ppg.timeout'))
            ->acceptJson()
            // Retry only the statuses that mean "not started, try again", and
            // only where a retry is safe. Deliberately NOT ConnectionException:
            // a connection that dies mid-POST may well have been received, and
            // an unseeded /v1/avatars call is not idempotent - retrying it
            // generates a second, different face and bills you the GPU time
            // twice. Seeded requests are content-addressed and safe to repeat;
            // if you want retries on connection errors, send a `seed`.
            ->retry(3, 2000, function (\Throwable $e) {
                return $e instanceof RequestException
                    && in_array($e->response->status(), [429, 502, 503, 504], true);
            }, throw: false)
            ->when(config('ppg.key'), fn (PendingRequest $r) => $r->withToken(config('ppg.key')));
    }

    /** @return array<string, mixed> */
    private function awaitJob(string $jobId, int $tries = 60): array
    {
        for ($i = 0; $i < $tries; $i++) {
            sleep(2);
            $job = $this->job($jobId);

            if ($job['status'] === 'done') {
                return $this->ok($this->http()->get("/v1/jobs/{$jobId}/results"))->json()[0];
            }

            if ($job['status'] === 'failed') {
                throw new RuntimeException("ppg job {$jobId} failed: " . ($job['error'] ?? 'unknown'));
            }
        }

        throw new RuntimeException("ppg job {$jobId} did not finish in time.");
    }

    private function ok(Response $response): Response
    {
        if ($response->successful()) {
            return $response;
        }

        // 422 carries a readable explanation from the safety filter.
        throw new RuntimeException(sprintf(
            'ppg %s: %s',
            $response->status(),
            $response->json('detail') ?? mb_substr($response->body(), 0, 300),
        ));
    }
}
```

## In a Blade template

If the generator is reachable from the browser (a private network, or behind
your own reverse proxy) this is the whole integration:

```blade
<img src="{{ config('ppg.url') }}/v1/avatars/by-seed/{{ app(App\Services\AvatarClient::class)->keyFor($user->email) }}?size=96"
     width="48" height="48" alt="" loading="lazy" class="rounded-full">
```

Route it through `keyFor()` rather than hashing inline: a bare `md5($email)` in
the URL is an email address anyone can recover from a log line.

The first request renders the face and blocks; every later request is a static
file read served with `Cache-Control: immutable`.

**With `PPG_API_KEY` set, an `<img>` tag will not work** — a browser cannot
send a bearer token. Proxy it through Laravel instead:

```php
// routes/web.php
Route::get('/avatar/{key}', function (string $key, AvatarClient $ppg) {
    $bytes = Cache::remember("avatar:{$key}:96", now()->addDays(30),
        fn () => $ppg->fetch($key, 96));

    return response($bytes, 200, [
        'Content-Type' => 'image/webp',
        'Cache-Control' => 'public, max-age=31536000, immutable',
    ]);
})->name('avatar');
```

```blade
<img src="{{ route('avatar', app(AvatarClient::class)->keyFor($user->email)) }}"
     width="48" height="48" alt="" loading="lazy">
```

## Bulk seeding with a queued job

`app/Jobs/GenerateAvatar.php`:

```php
<?php

namespace App\Jobs;

use App\Models\User;
use App\Services\AvatarClient;
use Illuminate\Contracts\Queue\ShouldBeUnique;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Queue\Queueable;

class GenerateAvatar implements ShouldQueue, ShouldBeUnique
{
    use Queueable;

    public int $tries = 3;
    public int $timeout = 300;

    // The generator has one worker and one GPU; there is nothing to gain from
    // hammering it. Run this queue with a single worker process.
    public array $backoff = [10, 60];

    public function __construct(public User $user) {}

    public function uniqueId(): string
    {
        return (string) $this->user->id;
    }

    public function handle(AvatarClient $ppg): void
    {
        $key = $ppg->keyFor($this->user->email);
        $result = $ppg->generate(seed: $key);

        $this->user->forceFill([
            'avatar_id' => $result['id'],
            'avatar_seed_key' => $key,
            'avatar_persona' => $result['persona'],
        ])->save();
    }
}
```

Dispatch it for everyone who has no avatar yet:

```php
User::whereNull('avatar_id')->chunkById(200, function ($users) {
    $users->each(fn (User $user) => GenerateAvatar::dispatch($user));
});
```

Migration for the three columns:

```php
Schema::table('users', function (Blueprint $table) {
    $table->string('avatar_id', 32)->nullable()->index();
    $table->string('avatar_seed_key', 64)->nullable();
    $table->json('avatar_persona')->nullable();
});
```

Storing `avatar_id` is optional — `by-seed` works without it — but it lets you
serve `/v1/avatars/{id}/image` directly and skip the seed lookup.

## A UserFactory that uses the persona

The generator already invents a coherent fictional person: name, age,
occupation and city that match the face. Using that instead of Faker means the
name in your seeded database matches the avatar next to it.

```php
<?php

namespace Database\Factories;

use App\Services\AvatarClient;
use Illuminate\Database\Eloquent\Factories\Factory;
use Illuminate\Support\Facades\Hash;
use Illuminate\Support\Str;

class UserFactory extends Factory
{
    /** One sequence number per created user, so seeds are stable across runs. */
    private static int $counter = 0;

    public function definition(): array
    {
        $seed = 'seed-user-' . (++self::$counter);
        $result = app(AvatarClient::class)->generate(seed: $seed);

        $persona = $result['persona'] ?? [];
        $name = $persona['name'] ?? $this->faker->name();

        return [
            'name' => $name,
            'email' => Str::slug($name, '.') . '.' . self::$counter . '@example.com',
            'email_verified_at' => now(),
            'password' => Hash::make('password'),
            'remember_token' => Str::random(10),

            'age' => $persona['age'] ?? null,
            'occupation' => $persona['occupation'] ?? null,
            'city' => $persona['city'] ?? null,
            'bio' => $persona['bio'] ?? null,

            'avatar_id' => $result['id'],
            'avatar_seed_key' => $seed,
        ];
    }
}
```

Two practical notes:

- **It is slow the first time.** Roughly 8 seconds per user on a 12 GB card, so
  `User::factory(50)->create()` is a coffee break. Because the seeds are
  deterministic (`seed-user-1`, `seed-user-2`, ...), the *second* run is served
  entirely from the generator's cache and takes milliseconds.
- **Warm the cache first for large seeds.** Generate them in one diverse batch,
  then run the factory:

  ```php
  app(AvatarClient::class)->batch(n: 200, seed: 'seed-users-2026');
  ```

  The batch endpoint spreads across sex, age and ancestry and avoids repeating
  combinations, which a loop of independent calls does not.

For tests, point `PPG_URL` at an instance running `PPG_BACKEND=fake`, or fake
the HTTP layer outright:

```php
Http::fake([
    '*/v1/avatars*' => Http::response([
        'id' => str_repeat('a', 32),
        'persona' => ['name' => 'Test Person', 'age' => 33, 'occupation' => 'baker', 'city' => 'Bergen'],
        'urls' => ['default' => '/v1/avatars/' . str_repeat('a', 32) . '/image'],
    ]),
]);
```

## The OpenAI-compatible route

If you already use `openai-php/laravel`, point its base URI at this service and
call `images()->create()`. It works, but the mapping is lossy: the prompt
becomes extra styling on top of a randomly sampled person rather than a
description of one. `POST /v1/avatars` is the better API — see
[API.md](API.md#post-v1imagesgenerations-openai-compatible).
