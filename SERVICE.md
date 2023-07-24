## Notes

```bash
curl http://127.0.0.1:5000/v1/completions -v -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI_API_KEY" --data "{\"model\":\"alpaca-lora-7b\",\"prompt\":\"Say this is a test\",\"max_tokens\":7,\"temperature\":0}"
*   Trying 127.0.0.1:5000...
* Connected to 127.0.0.1 (127.0.0.1) port 5000 (#0)
> POST /v1/completions HTTP/1.1
> Host: 127.0.0.1:5000
> User-Agent: curl/7.83.1
> Accept: */*
> Content-Type: application/json
> Authorization: Bearer $OPENAI_API_KEY
> Content-Length: 87
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Server: Werkzeug/2.2.3 Python/3.10.9
< Date: Fri, 24 Mar 2023 22:19:13 GMT
< Content-Type: application/json
< Content-Length: 226
< Connection: close
<
{"choices":[{"finish_reason":"length","text":"This is a test."}],"created":1679696353,"id":"dummy","model":"tloen/alpaca-lora-7b","object":"text_completion","usage":{"completion_tokens":6,"prompt_tokens":6,"total_tokens":12}}
* Closing connection 0
curl http://127.0.0.1:5000/v1/chat/completions -v -H "Content-Type: application/json" -H "Authorization: Bearer $OPENAI_API_KEY" --data "{\"model\":\"alpaca-lora-7b\",\"max_tokens\":64,\"temperature\":0.95,  \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}"
*   Trying 127.0.0.1:5000...
* Connected to 127.0.0.1 (127.0.0.1) port 5000 (#0)
> POST /v1/chat/completions HTTP/1.1
> Host: 127.0.0.1:5000
> User-Agent: curl/7.83.1
> Accept: */*
> Content-Type: application/json
> Authorization: Bearer $OPENAI_API_KEY
> Content-Length: 115
>
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Server: Werkzeug/2.2.3 Python/3.10.9
< Date: Fri, 24 Mar 2023 22:25:01 GMT
< Content-Type: application/json
< Content-Length: 257
< Connection: close
<
{"choices":[{"content":"Hello! How can I help you?","finish_reason":"stop","role":"assistant"}],"created":1679696701,"id":"dummy","model":"tloen/alpaca-lora-7b","object":"text_completion","usage":{"completion_tokens":9,"prompt_tokens":6,"total_tokens":15}}
* Closing connection 0
curl http://127.0.0.1:5000/v1/models
```

```python
llamaModels = [
    'llama-7b-hf',
    'alpaca-7b-hf',
    'decapoda-research/llama-7b-hf',
    'tloen/alpaca-lora-7b',
    'decapoda-research/llama-7b-hf-int4',
    'decapoda-research/llama-13b-hf-int4',
    'decapoda-research/llama-65b-hf-int4',
    'decapoda-research/llama-30b-hf-int4',
    'decapoda-research/llama-30b-hf',
    'decapoda-research/llama-65b-hf',
    'decapoda-research/llama-13b-hf',
    'decapoda-research/llama-smallint-pt',
    'decapoda-research/llama-7b-hf-int8',
]
```