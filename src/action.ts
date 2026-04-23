// correction-reactor — when an upstream Observation is revised, find all
// AttributionBelief assertions that cited it and re-run the originating
// persona's LLM to produce revised beliefs.
//
// Trigger: revise operation on Observation in noaa-sst-daily or usgs-nwis.
// Scope : chesapeake-attribution (home repo for beliefs).

import OpenAI from "openai";
import type { Operation, ReviseOperation, FilterResult, ThingGet } from "@warmhub/sdk-ts";
import { clientFromEnv, homeRepo, splitRepo } from "./warmhub";

const MODEL = process.env.PERSONA_MODEL ?? "anthropic/claude-3.5-sonnet";
const OPENROUTER_BASE = process.env.OPENROUTER_BASE ?? "https://openrouter.ai/api/v1";
const CONTEXT_LIMIT = 40;
const RADIUS_DEG = 0.75;
const DAYS_BEFORE = 14;

const CAUSES = ["agricultural","thermal","industrial","stormflow","biological","unknown"] as const;
type Cause = (typeof CAUSES)[number];
type Persona = "ngo" | "industry" | "agency" | "academic";

const UPSTREAM_REPOS = {
  noaa: "fish-kill-attribution/noaa-sst-daily",
  usgs: "fish-kill-attribution/usgs-nwis",
  epa:  "fish-kill-attribution/epa-tri",
  reports: "fish-kill-attribution/state-fishkills",
} as const;

const PERSONA_PRIORS: Record<Persona, string> = {
  ngo: `Your priors favor causes linked to industrial pollution, agricultural runoff, stormwater discharge, and regulatory failures. You weigh EPA TRI releases and nutrient loading heavily.`,
  industry: `Your priors favor natural and climatic causes — harmful algal blooms, thermal stress, hypoxia, fish disease, and stormflow. You are skeptical of facility attribution absent strong contemporaneous exceedances.`,
  agency: `Your priors are balanced and method-conservative. You weigh regulatory compliance history, environmental conditions, and prior adjudicated cases equally.`,
  academic: `Your priors favor methodologically-validated causation. You weight multiple independent data streams and quantitative consistency.`,
};

interface MatchedOp { name?: string; kind?: string; operation?: string; wref?: string }
interface SpritePayload { matchedOperations?: MatchedOp[]; sourceRepo?: string }
interface EventData {
  lat: number; lon: number; date: string;
  watershed?: string; location_name?: string;
  primary_species?: string | null;
  estimated_mortality?: number;
}
interface ContextItem { wref: string; summary: string }
interface LlmBelief {
  cause: Cause;
  share: number;
  rationale: string;
  sl_belief: number;
  sl_disbelief: number;
  sl_uncertainty: number;
  sl_base_rate?: number;
  evidence_ids?: string[];
}

function clamp01(n: number): number {
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}
function normalizeSl(b: LlmBelief): LlmBelief {
  const belief = clamp01(b.sl_belief);
  const disbelief = clamp01(b.sl_disbelief);
  const uncertainty = clamp01(b.sl_uncertainty);
  const sum = belief + disbelief + uncertainty;
  if (sum === 0) return { ...b, sl_belief: 0, sl_disbelief: 0, sl_uncertainty: 1 };
  return {
    ...b,
    sl_belief: belief / sum,
    sl_disbelief: disbelief / sum,
    sl_uncertainty: uncertainty / sum,
    sl_base_rate: clamp01(b.sl_base_rate ?? 0.5),
    share: clamp01(b.share),
  };
}
function toDateMs(iso: string | undefined): number {
  if (!iso) return 0;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? t : 0;
}
function degDistance(a: { lat: number; lon: number }, b: { lat: number; lon: number }): number {
  return Math.sqrt((a.lat - b.lat) ** 2 + (a.lon - b.lon) ** 2);
}

// Build a full qualified wref for a thing in another repo.
function qualifyWref(repo: string, shape: string, name: string): string {
  return `wh:${repo}/${shape}/${name}`;
}

// Parse the payload to find the revised Observation wref, qualified.
function revisedObservationWref(payload: SpritePayload): string | null {
  const sourceRepo = payload.sourceRepo ?? "";
  const ops = payload.matchedOperations ?? [];
  for (const op of ops) {
    if (op.operation !== "revise") continue;
    const name = op.name ?? op.wref ?? "";
    if (!name) continue;
    // If already qualified (wh:org/repo/Shape/name), use as-is.
    if (name.startsWith("wh:")) return name;
    // Otherwise qualify with sourceRepo if available.
    if (sourceRepo) return `wh:${sourceRepo}/${name}`;
    // Fallback: name alone — can't qualify.
    return name;
  }
  return null;
}

async function fetchAllBeliefs(client: ReturnType<typeof clientFromEnv>, org: string, repo: string) {
  const all: any[] = [];
  let cursor: string | undefined;
  for (let i = 0; i < 20; i++) {
    const page: FilterResult = await client.thing.query(org, repo, {
      shape: "AttributionBelief",
      limit: 500,
      cursor,
    });
    const items = (page.items ?? []) as any[];
    all.push(...items);
    cursor = (page as any).nextCursor ?? (page as any).cursor;
    if (!cursor || items.length === 0) break;
  }
  return all;
}

async function queryNearby(
  client: ReturnType<typeof clientFromEnv>,
  repo: string,
  shape: string,
  event: EventData,
): Promise<ContextItem[]> {
  const { orgName, repoName } = splitRepo(repo);
  let result: FilterResult;
  try {
    result = await client.thing.query(orgName, repoName, { shape, limit: CONTEXT_LIMIT * 3 });
  } catch (err) {
    console.error(`query ${repo}/${shape} failed:`, (err as Error).message);
    return [];
  }

  const eventMs = toDateMs(event.date);
  const windowMs = DAYS_BEFORE * 86_400_000;
  const picks: ContextItem[] = [];

  for (const row of result.items ?? []) {
    const data = (row as any).data ?? (row as any).head?.data ?? {};
    const lat = Number(data.lat);
    const lon = Number(data.lon);
    if (Number.isFinite(lat) && Number.isFinite(lon)) {
      if (degDistance({ lat, lon }, event) > RADIUS_DEG) continue;
    }
    const ts = data.timestamp ?? data.date ?? null;
    if (ts) {
      const tMs = toDateMs(ts);
      if (tMs && eventMs && (tMs > eventMs || eventMs - tMs > windowMs)) continue;
    }
    picks.push({
      wref: qualifyWref(repo, shape, row.name),
      summary: JSON.stringify({ name: row.name, ...Object.fromEntries(Object.entries(data).slice(0, 10)) }),
    });
    if (picks.length >= CONTEXT_LIMIT) break;
  }
  return picks;
}

async function gatherContext(client: ReturnType<typeof clientFromEnv>, event: EventData): Promise<ContextItem[]> {
  const [o1, o2, r, s] = await Promise.all([
    queryNearby(client, UPSTREAM_REPOS.noaa, "Observation", event),
    queryNearby(client, UPSTREAM_REPOS.usgs, "Observation", event),
    queryNearby(client, UPSTREAM_REPOS.epa, "Release", event),
    queryNearby(client, UPSTREAM_REPOS.reports, "FishKillReport", event),
  ]);
  return [...o1, ...o2, ...r, ...s];
}

function systemPrompt(persona: Persona): string {
  return `You are an attribution analyst with ${persona} priors.
${PERSONA_PRIORS[persona]}

Given a fish-kill event and nearby upstream observations, releases, and prior
reports, produce causal attribution beliefs across these causes:
${CAUSES.join(", ")}.

For each cause you assign non-zero share, produce a Subjective Logic opinion
(sl_belief + sl_disbelief + sl_uncertainty = 1.0) and cite concrete evidence
wrefs drawn from the provided context.

Output STRICT JSON:
{ "beliefs": [ { "cause": "...", "share": 0..1, "rationale": "...", "sl_belief": 0..1, "sl_disbelief": 0..1, "sl_uncertainty": 0..1, "sl_base_rate": 0..1, "evidence_ids": ["..."] } ] }

Only include causes whose share > 0. Shares should sum to ~1.0. Do not invent wrefs.`;
}

async function askLlm(openai: OpenAI, persona: Persona, event: EventData, context: ContextItem[]): Promise<LlmBelief[]> {
  const res = await openai.chat.completions.create({
    model: MODEL,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: systemPrompt(persona) },
      { role: "user",   content: JSON.stringify({ event, context: context.map((c) => ({ wref: c.wref, data: c.summary })) }, null, 2) },
    ],
  });
  const raw = res.choices[0]?.message?.content ?? "{}";
  const parsed = JSON.parse(raw) as { beliefs?: LlmBelief[] };
  const beliefs = Array.isArray(parsed.beliefs) ? parsed.beliefs : [];
  return beliefs
    .filter((b) => CAUSES.includes(b.cause))
    .map(normalizeSl)
    .filter((b) => b.share > 0);
}

async function main() {
  const client = clientFromEnv();
  const { orgName, repoName } = splitRepo(homeRepo()); // chesapeake-attribution

  const raw = await Bun.stdin.text();
  const payload: SpritePayload = raw ? JSON.parse(raw) : {};
  const revisedWref = revisedObservationWref(payload);
  if (!revisedWref) {
    console.log(JSON.stringify({ skipped: true, reason: "no revise op in payload" }));
    return;
  }

  // Find affected beliefs: fetch all AttributionBelief HEADs, filter client-side by evidence_ids.
  const allBeliefs = await fetchAllBeliefs(client, orgName, repoName);
  const affected = allBeliefs.filter((b) => {
    const ev = (b.data?.evidence_ids ?? b.head?.data?.evidence_ids ?? []) as string[];
    return Array.isArray(ev) && ev.some((id) => id === revisedWref || id.endsWith(revisedWref));
  });
  if (affected.length === 0) {
    console.log(JSON.stringify({ revisedWref, affected: 0, note: "no beliefs cite this observation" }));
    return;
  }

  // Group affected beliefs by (event, persona) so we re-run the LLM once per group.
  const groups = new Map<string, { eventWref: string; persona: Persona; beliefNames: string[] }>();
  for (const b of affected) {
    const data = b.data ?? b.head?.data ?? {};
    const about = (b.about ?? b.head?.about ?? "") as string;
    const persona = (data.persona ?? "") as Persona;
    if (!about || !persona) continue;
    const key = `${about}::${persona}`;
    let g = groups.get(key);
    if (!g) { g = { eventWref: about, persona, beliefNames: [] }; groups.set(key, g); }
    g.beliefNames.push(b.name);
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error("OPENROUTER_API_KEY not set (credential binding?)");
  const openai = new OpenAI({
    apiKey,
    baseURL: OPENROUTER_BASE,
    defaultHeaders: {
      "HTTP-Referer": "https://github.com/ntoft/correction-reactor",
      "X-Title": "fish-kill-attribution-correction-reactor",
    },
  });

  const summary: any[] = [];
  for (const group of groups.values()) {
    const eventThing = await client.thing.get(orgName, repoName, group.eventWref) as ThingGet;
    const eventData = ((eventThing as any).head?.data ?? (eventThing as any).data ?? {}) as EventData;
    if (!eventData?.date || eventData.lat == null) {
      summary.push({ eventWref: group.eventWref, persona: group.persona, status: "event missing data" });
      continue;
    }

    const context = await gatherContext(client, eventData);
    const newBeliefs = await askLlm(openai, group.persona, eventData, context);
    const byCause = new Map<Cause, LlmBelief>();
    for (const nb of newBeliefs) byCause.set(nb.cause, nb);

    // Revise each affected belief with the new corresponding cause opinion.
    // Belief names follow: AttributionBelief/<event-slug>-<persona>-<cause>
    const ops: Operation[] = [];
    for (const name of group.beliefNames) {
      const m = name.match(/-(agricultural|thermal|industrial|stormflow|biological|unknown)$/);
      const cause = (m?.[1] ?? "unknown") as Cause;
      const nb = byCause.get(cause);
      if (!nb) {
        // LLM no longer believes in this cause — zero it out rather than retract.
        const revise: ReviseOperation = {
          operation: "revise",
          kind: "assertion",
          name,
          data: {
            share: 0,
            sl_belief: 0,
            sl_disbelief: 1,
            sl_uncertainty: 0,
            rationale: "Revised: upstream evidence changed; this cause no longer supported by persona.",
          },
        };
        ops.push(revise);
        continue;
      }
      const revise: ReviseOperation = {
        operation: "revise",
        kind: "assertion",
        name,
        data: {
          share: nb.share,
          sl_belief: nb.sl_belief,
          sl_disbelief: nb.sl_disbelief,
          sl_uncertainty: nb.sl_uncertainty,
          sl_base_rate: nb.sl_base_rate ?? 0.5,
          rationale: nb.rationale,
          evidence_ids: nb.evidence_ids ?? [],
          model: MODEL,
        },
      };
      ops.push(revise);
    }

    const result = await client.commit.apply(
      orgName, repoName,
      `correction-reactor: revised ${group.persona} beliefs for ${group.eventWref} (trigger=${revisedWref})`,
      ops,
    );
    summary.push({
      eventWref: group.eventWref,
      persona: group.persona,
      commitId: result.commitId,
      revisedBeliefs: ops.length,
    });
  }

  console.log(JSON.stringify({ revisedWref, affectedBeliefs: affected.length, groups: groups.size, results: summary }));
}

main().catch((err) => { console.error(err); process.exit(1); });
