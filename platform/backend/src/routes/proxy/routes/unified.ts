import { RouteId } from "@shared";
import type { FastifyPluginAsyncZod } from "fastify-type-provider-zod";
import { z } from "zod";
import logger from "@/logging";
import {
  OpenAI,
  constructResponseSchema,
  type SupportedProvider,
} from "@/types";
import {
  openaiAdapterFactory,
  anthropicAdapterFactory,
  geminiAdapterFactory,
  cohereAdapterFactory,
  bedrockAdapterFactory,
} from "../adapters";
import { PROXY_API_PREFIX, PROXY_BODY_LIMIT } from "../common";
import { handleLLMProxy } from "../llm-proxy-handler";

const UNIFIED_PREFIX = `${PROXY_API_PREFIX}/unified`;
const CHAT_SUFFIX = "/chat/completions";
const MODELS_SUFFIX = "/models";

type AdapterFactory = typeof openaiAdapterFactory;

/**
 * Resolve the right provider adapter for a given model name.
 * Order matters: more-specific prefixes first.
 *
 * DRAFT v1 mapping. See PR description open questions #1 and #2 for
 * the conventions we still need maintainer input on.
 */
export function resolveAdapterForModel(model: string): {
  factory: AdapterFactory;
  provider: SupportedProvider;
} {
  // Bedrock-prefixed models go to bedrock first.
  if (
    model.startsWith("anthropic.") ||
    model.startsWith("amazon.") ||
    model.startsWith("meta.") ||
    model.startsWith("mistral.") ||
    model.startsWith("cohere.command-r")
  ) {
    return { factory: bedrockAdapterFactory, provider: "bedrock" };
  }
  if (model.startsWith("claude-")) {
    return { factory: anthropicAdapterFactory, provider: "anthropic" };
  }
  if (model.startsWith("gemini-")) {
    return { factory: geminiAdapterFactory, provider: "gemini" };
  }
  if (model.startsWith("command")) {
    return { factory: cohereAdapterFactory, provider: "cohere" };
  }
  if (
    model.startsWith("gpt-") ||
    model.startsWith("o1-") ||
    model.startsWith("o3-") ||
    model.startsWith("o4-") ||
    model === "chatgpt-4o-latest"
  ) {
    return { factory: openaiAdapterFactory, provider: "openai" };
  }
  throw new Error(
    `Unknown model "${model}". Unified endpoint supports models prefixed with: gpt-, o1-, o3-, o4-, claude-, gemini-, command-, anthropic.*, amazon.*, meta.*.`,
  );
}

/**
 * Unified LLM proxy routes.
 *
 * Exposes /v1/unified/v1/chat/completions and /v1/unified/v1/models that accept
 * OpenAI-format requests, route to the matching upstream provider based on the
 * model name, and return OpenAI-format responses.
 *
 * Reuses handleLLMProxy + existing per-provider adapter factories so all
 * existing virtual-key, auth, observability, and security-policy plumbing
 * applies unchanged.
 */
const unifiedProxyRoutes: FastifyPluginAsyncZod = async (fastify) => {
  logger.info("[UnifiedProxy] Registering unified LLM proxy routes");

  fastify.post(
    `${UNIFIED_PREFIX}/v1${CHAT_SUFFIX}`,
    {
      bodyLimit: PROXY_BODY_LIMIT,
      schema: {
        operationId: RouteId.UnifiedChatCompletions,
        description:
          "Send an OpenAI-format chat completion that is routed to the upstream provider matching the requested model.",
        tags: ["LLM Proxy"],
        body: OpenAI.API.ChatCompletionsRequestSchema,
        response: constructResponseSchema(
          OpenAI.API.ChatCompletionsResponseSchema,
        ),
      },
    },
    async (request, reply) => {
      const body = request.body as { model: string };
      let resolved: ReturnType<typeof resolveAdapterForModel>;
      try {
        resolved = resolveAdapterForModel(body.model);
      } catch (e) {
        return reply.status(400).send({
          error: {
            type: "invalid_model",
            message: (e as Error).message,
          },
        });
      }
      logger.info(
        { model: body.model, provider: resolved.provider },
        "[UnifiedProxy] Routing request",
      );
      return handleLLMProxy(request.body, request, reply, resolved.factory);
    },
  );

  fastify.get(
    `${UNIFIED_PREFIX}/v1${MODELS_SUFFIX}`,
    {
      schema: {
        operationId: RouteId.UnifiedListModels,
        description:
          "List all models from all configured providers in OpenAI list format.",
        tags: ["LLM Proxy"],
        response: {
          200: z.object({
            object: z.literal("list"),
            data: z.array(
              z.object({
                id: z.string(),
                object: z.literal("model"),
                created: z.number(),
                owned_by: z.string(),
              }),
            ),
          }),
        },
      },
    },
    async (_request, reply) => {
      // TODO (PR description Q4): aggregate from each configured provider's
      // /v1/models endpoint and tag each model with its upstream provider.
      // Skeleton returns an empty list; full implementation lands after
      // maintainer confirms the open questions.
      return reply.send({ object: "list" as const, data: [] });
    },
  );
};

export default unifiedProxyRoutes;
