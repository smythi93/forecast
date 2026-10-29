"""
FORECAST's multi-turn model interface: pricing, budgets, and error types.

Runs offline: litellm's ``mock_response`` stands in for the provider.
"""

import os
import unittest
from unittest import mock

import litellm

from forecast.agents.models.anthropic import AnthropicClient
from forecast.agents.models.ollama import OllamaClient
from forecast.agents.models.utils import PRICES, THINKING_TOKENS, price

SONNET = "claude-sonnet-5-5"
SONNET_4 = "claude-sonnet-4-20250514"


class PricingTests(unittest.TestCase):
    def test_sonnet_list_prices(self):
        self.assertAlmostEqual(12.0, price(SONNET, 1_000_000, 1_000_000))
        self.assertAlmostEqual(18.0, price(SONNET_4, 1_000_000, 1_000_000))

    def test_cached_prompt_tokens_are_charged_at_cache_rates(self):
        # 1M prompt tokens of which 400k written to and 500k read from cache.
        expected = 100_000 * 2e-6 + 400_000 * 2.5e-6 + 500_000 * 0.2e-6
        self.assertAlmostEqual(expected, price(SONNET, 1_000_000, 0, 400_000, 500_000))

    def test_unknown_model_costs_nothing_and_provider_prefix_is_ignored(self):
        self.assertEqual(0.0, price("qwen3.6:35b", 1000, 1000))
        self.assertEqual(price(SONNET, 10, 10), price(f"anthropic/{SONNET}", 10, 10))

    def test_price_table_does_not_depend_on_litellm(self):
        # litellm has dropped Sonnet 4 from its table; ours must not.
        self.assertIn(SONNET, PRICES)
        self.assertIn(SONNET_4, PRICES)

    def test_haiku_list_price(self):
        self.assertAlmostEqual(6.0, price("claude-haiku-4-5-20251001", 1_000_000, 1_000_000))


class ChatTests(unittest.TestCase):
    def test_chat_charges_the_clients_own_prices(self):
        reply = AnthropicClient(model=SONNET, api_key="x").chat(
            [{"role": "user", "content": "hi"}], mock_response="hello"
        )[0]
        self.assertEqual("hello", reply.content)
        self.assertAlmostEqual(
            price(SONNET, reply.prompt_tokens, reply.completion_tokens), reply.cost
        )
        self.assertGreater(reply.cost, 0)

    def test_local_model_is_priced_as_the_compared_model(self):
        with mock.patch.dict(os.environ, {"FORECAST_PRICE_AS": SONNET}):
            client = OllamaClient(model="qwen3.6:35b")
        reply = client.chat([{"role": "user", "content": "hi"}], mock_response="hello")[0]
        self.assertGreater(reply.cost, 0)
        self.assertEqual(client.num_ctx, client.context_window)

    def test_provider_exception_types_reach_the_agent(self):
        client = AnthropicClient(model=SONNET, api_key="x")
        error = litellm.ContextWindowExceededError(
            message="too long", model=SONNET, llm_provider="anthropic"
        )
        with self.assertRaises(litellm.ContextWindowExceededError):
            client.chat([{"role": "user", "content": "hi"}], mock_response=error)

    def test_recorder_sees_every_call(self):
        client = AnthropicClient(model=SONNET, api_key="x")
        records = []
        client.recorder = records.append
        client.chat([{"role": "user", "content": "hi"}], mock_response="a", n=2)
        self.assertEqual(2, len(records))
        self.assertTrue(all(r["cost"] > 0 for r in records))

    def test_model_without_sampling_gets_none_and_room_to_think(self):
        client = AnthropicClient(model=SONNET, api_key="x")
        records = []
        client.recorder = records.append
        with mock.patch("litellm.completion", wraps=litellm.completion) as completion:
            client.chat([{"role": "user", "content": "hi"}], temperature=0.2, top_p=1.0,
                        max_tokens=1024, mock_response="a")
        sent = completion.call_args.kwargs
        self.assertNotIn("temperature", sent)
        self.assertNotIn("top_p", sent)
        self.assertEqual(1024 + THINKING_TOKENS, sent["max_tokens"])
        self.assertEqual((None, 0.2), (records[0]["temperature"], records[0]["requested_temperature"]))
        self.assertEqual(1024, records[0]["requested_max_tokens"])

    def test_thinking_room_stays_within_the_models_output_limit(self):
        client = AnthropicClient(model=SONNET, api_key="x")
        with mock.patch("litellm.completion", wraps=litellm.completion) as completion:
            client.chat([{"role": "user", "content": "hi"}], max_tokens=128_000, mock_response="a")
        self.assertEqual(128_000, completion.call_args.kwargs["max_tokens"])

    def test_claude_4_5_gets_the_temperature_without_top_p(self):
        client = AnthropicClient(model="claude-haiku-4-5-20251001", api_key="x")
        with mock.patch("litellm.completion", wraps=litellm.completion) as completion:
            client.chat([{"role": "user", "content": "hi"}], temperature=0.0, top_p=1.0,
                        mock_response="a")
        sent = completion.call_args.kwargs
        self.assertEqual(0.0, sent["temperature"])
        self.assertNotIn("top_p", sent)

    def test_model_with_sampling_gets_the_agents_settings(self):
        client = AnthropicClient(model=SONNET_4, api_key="x")
        with mock.patch("litellm.completion", wraps=litellm.completion) as completion:
            client.chat([{"role": "user", "content": "hi"}], temperature=0.2, max_tokens=1024,
                        mock_response="a")
        sent = completion.call_args.kwargs
        self.assertEqual((0.2, 1024), (sent["temperature"], sent["max_tokens"]))


if __name__ == "__main__":
    unittest.main()


class ProxyTests(unittest.TestCase):
    ENV = {"LITELLM_PROXY_API_BASE": "http://proxy.invalid", "LITELLM_PROXY_API_KEY": "k",
           "LITELLM_PROXY_CANONICAL_MODEL": SONNET}

    def test_proxy_alias_routes_through_litellm_proxy_and_prices_canonically(self):
        from forecast.agents.models.utils import get_model_client

        with mock.patch.dict(os.environ, self.ENV):
            client = get_model_client("proxy/cispa-claude-sonnet")
        self.assertEqual("litellm_proxy/cispa-claude-sonnet", client.litellm_model_name())
        self.assertEqual(SONNET, client.model)
        self.assertEqual(1_000_000, client.context_window)
        reply = client.chat([{"role": "user", "content": "hi"}], mock_response="hello")[0]
        self.assertAlmostEqual(price(SONNET, reply.prompt_tokens, reply.completion_tokens), reply.cost)

    def test_swe_agent_gets_the_canonical_models_defaults(self):
        try:
            from forecast.agents.swe_agent import SWEAgentRepairAgent
        except ImportError:  # pragma: no cover
            self.skipTest("SWE-agent not installed")
        from forecast.agents.models.proxy import LitellmProxyClient

        with mock.patch.dict(os.environ, self.ENV):
            client = LitellmProxyClient("proxy/cispa-claude-sonnet")
        agent = SWEAgentRepairAgent().make_agent(client)
        self.assertEqual(SONNET, agent.model.config.name)
        self.assertEqual(128_000, agent.model.model_max_output_tokens)
        self.assertEqual(1_000_000 - 128_000, agent.model.model_max_input_tokens)
