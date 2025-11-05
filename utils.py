import os
import asyncio
from safetytooling.apis import InferenceAPI
from safetytooling.data_models import Prompt, LLMResponse
from pathlib import Path

os.environ["OPENAI_API_KEY"] = "dummy"
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "")

API = InferenceAPI(cache_dir=Path(os.getenv("OR_CACHE_DIR", "workspace/.cache/openrouter")), openrouter_num_threads=100)
semaphore = asyncio.Semaphore(100)

def get_few_shot_prompt(prompts_and_responses: list[tuple[str, str]]) -> list[dict]:
  messages = []
  for p, r in prompts_and_responses:
    messages.append(
        {
            "role": "user",
            "content": p,
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": r
        }
    )

  return messages

async def get_message_with_few_shot_prompt(
    few_shot_prompt: list[dict],
    prompt: str,
    system_prompt: str,
    model: str = "google/gemini-2.5-flash",
    max_retries: int = 5,
    max_tokens: int = 500,
    temperature: float = 0,
    verbose: bool = False,
    **kwargs
) -> LLMResponse:

    system_prompt = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    user_prompt = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    messages = system_prompt + few_shot_prompt + user_prompt
    prompt = Prompt(messages=messages)

    async with semaphore:

        responses = await API.__call__(
            model_id=model,
            prompt=prompt,
            max_attempts_per_api_call=max_retries,
            force_provider="openrouter",
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        response = responses[0]
        if verbose:
            print(f"Got response from {model} after {response.duration:.2f}s")

        return response

async def _get_messages_with_few_shot_prompt(
    few_shot_prompt: list[dict] | list[str],
    prompts: list[str],
    system_prompt: str,
    **kwargs
) -> list[LLMResponse]:
  messages = await asyncio.gather(
      *[
          get_message_with_few_shot_prompt(
              few_shot_prompt,
              prompt=p,
              system_prompt=system_prompt,
              **kwargs
          )
          for p in prompts
      ]
  )
  return messages

def get_messages_with_few_shot_prompt(
    few_shot_prompt: list[dict] | list[str],
    prompts: list[str],
    system_prompt: str,
    **kwargs
) -> list[LLMResponse]:
  return asyncio.run(
      _get_messages_with_few_shot_prompt(
          few_shot_prompt,
          prompts,
          system_prompt,
          **kwargs
      )
  )

if __name__ == "__main__":
    system_prompt = "You are a math expert and you solve problems. Just answer with the final answer."
    few_shot_prompt = get_few_shot_prompt([("2 + 2", "4"), ("49 * 7", "343"), ("12 / 4", "3"), ("15 - 6", "9"), ("8 ** 2", "64")])
    messages = get_messages_with_few_shot_prompt(few_shot_prompt, ["64 ** 2", "243 / 7", "999 * 8"], system_prompt=system_prompt, model="google/gemini-2.5-flash")

    for msg in messages:
        print(msg.completion)