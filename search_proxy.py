from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from duckduckgo_search import DDGS
import requests
import uvicorn
import json

app = FastAPI()

# Point this to your REAL llama.cpp server
LLAMA_API_URL = "http://localhost:8012/v1/completions"


def search_web(query):
    print(f"🔎 Searching web for: {query}")
    try:
        results = DDGS().text(query, max_results=3)
        if not results:
            return ""
        context = "\n# [WEB SEARCH RESULTS]:\n"
        for r in results:
            context += f"# - {r['title']}: {r['body']}\n"
        context += "# [END SEARCH RESULTS]\n\n"
        return context
    except Exception as e:
        print(f"Search failed: {e}")
        return ""


async def process_request(request: Request):
    try:
        data = await request.json()

        # 1. CONVERT CHAT TO PROMPT
        # Minuet sends "messages", we need "prompt" for Llama
        if "prompt" not in data and "messages" in data:
            messages = data["messages"]
            if messages:
                data["prompt"] = messages[-1]["content"]
                data.pop("messages", None)
                data.pop("model", None)

        prompt = data.get("prompt", "")

        # 2. PERFORM SEARCH (If needed)
        if "### SEARCH" in prompt or (
            len(prompt) < 300 and "?" in prompt and "#" in prompt
        ):
            if "### SEARCH" in prompt:
                query = prompt.split("### SEARCH")[1].split("\n")[0].strip()
            else:
                query = prompt.strip().split("\n")[-1].replace("#", "").strip()

            if len(query) > 3:
                search_context = search_web(query)
                if search_context:
                    data["prompt"] = search_context + prompt

        # 3. STREAM & TRANSLATE RESPONSE
        def response_translator():
            # Force streaming
            data["stream"] = True
            try:
                with requests.post(LLAMA_API_URL, json=data, stream=True) as r:
                    for line in r.iter_lines():
                        if line:
                            decoded_line = line.decode("utf-8")

                            # Pass [DONE] signal straight through
                            if "[DONE]" in decoded_line:
                                yield decoded_line + "\n\n"
                                continue

                            # Parse the Llama "Completion" JSON
                            if decoded_line.startswith("data:"):
                                try:
                                    json_str = decoded_line.replace("data: ", "")
                                    chunk = json.loads(json_str)

                                    # Extract the text
                                    text = chunk["choices"][0].get("text", "")

                                    # REPACKAGE as "Chat" JSON for Neovim
                                    # Neovim expects: choices[0].delta.content
                                    new_chunk = {
                                        "id": "chatcmpl-proxy",
                                        "object": "chat.completion.chunk",
                                        "created": 0,
                                        "model": "gemma3-proxy",
                                        "choices": [
                                            {
                                                "index": 0,
                                                "delta": {"content": text},
                                                "finish_reason": None,
                                            }
                                        ],
                                    }

                                    # Send it back to Neovim
                                    yield "data: " + json.dumps(new_chunk) + "\n\n"
                                except:
                                    # If parsing fails, just send original (fallback)
                                    yield decoded_line + "\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(response_translator(), media_type="text/event-stream")

    except Exception as e:
        return {"error": str(e)}


# Listen on both ports
@app.post("/v1/completions")
async def handle_completion(request: Request):
    return await process_request(request)


@app.post("/v1/chat/completions")
async def handle_chat(request: Request):
    return await process_request(request)


if __name__ == "__main__":
    print("🚀 Translator Proxy running on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
