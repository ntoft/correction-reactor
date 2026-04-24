import { WarmHubClient } from "@warmhub/sdk-ts";

const DEFAULT_HOME_REPO = "fish-kill-attribution/chesapeake-attribution";

export function clientFromEnv(): WarmHubClient {
  // Runtime injects WH_TOKEN; api url is WARMHUB_API_URL.
  const apiUrl = process.env.WARMHUB_API_URL ?? "https://api.warmhub.ai";
  const token = process.env.WH_TOKEN ?? process.env.WARMHUB_TOKEN;
  if (!token) throw new Error("WH_TOKEN not set (sprite runtime should inject)");
  return new WarmHubClient({ apiUrl, accessToken: () => token });
}

export function homeRepo(): string {
  return process.env.WARMHUB_HOME_REPO ?? process.env.WARMHUB_REPO ?? DEFAULT_HOME_REPO;
}

export function splitRepo(repo: string): { orgName: string; repoName: string } {
  const [orgName, repoName] = repo.split("/");
  if (!orgName || !repoName) {
    throw new Error(`Expected repo in "org/name" form, got ${JSON.stringify(repo)}`);
  }
  return { orgName, repoName };
}


// Redeem the short-lived secretToken from the sprite payload against the
// credentials/export endpoint; merge returned secrets into process.env so
// existing reads like `process.env.OPENROUTER_API_KEY` work unchanged.
export async function loadCredentialsFromPayload(payload: any): Promise<void> {
  const token = payload?.secretToken ?? payload?.payload?.secretToken;
  if (!token) return;
  const apiUrl = process.env.WARMHUB_API_URL ?? "https://api.warmhub.ai";
  const resp = await fetch(`${apiUrl}/api/credentials/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });
  if (!resp.ok) {
    throw new Error(`credentials/export ${resp.status}: ${await resp.text()}`);
  }
  const { secrets } = (await resp.json()) as { secrets?: Record<string, string> };
  for (const [k, v] of Object.entries(secrets ?? {})) {
    if (process.env[k] === undefined) process.env[k] = v;
  }
}
