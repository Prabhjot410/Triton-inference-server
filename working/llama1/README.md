# 🦙 Triton Inference with LLaMA Python Backend

This repository demonstrates how to deploy a LLaMA-based model using NVIDIA Triton Inference Server with sequence batching support using the Python backend.

---

## 🚀 How to Run the Container

You can run the Triton server container with the mounted model repository as follows:

```bash
docker run --gpus=all --rm -it \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v/path/to/your/model/repo:/models \
  nvcr.io/nvidia/tritonserver:24.05-py3 \
  tritonserver --model-repository=/models
```

Replace `/path/to/your/model/repo` with the actual path to your Triton model directory containing `llama1`.

---

## 🧪 Running the Inference Client

After the server is up and running, run the Python test client:

```bash
python test_llama.py
```

---

## 📥 Example Output

```
[1] Prompt: Hello, my name is Prabhjot
    ➜ Response: Hello, my name is Prabhjot Singh from India, I would like to share with you about a new and innovative way of cooking called "Vegetable Curry"...
    (Full recipe follows...)

[2] Prompt: what is my name?
    ➜ Response: Hello, my name is Prabhjot
    what is my name?
    Prabhjot: It's Prabhjot, I'm a graphic designer

    Host: Great! Can you provide some information about a famous Indian poet, Rabindranath Tagore?

    Prabhjot: Rabindranath Tagore was born on November 18, 1861 in Kolkata...
    (Extended conversation continues...)

[3] Prompt: Tell me a joke.
    ➜ Response: Joke: A woman is in the grocery store, and she finds a candy bar with a sign that says "Most of us would rather die than lose our teeth."...
    (Multiple jokes returned with "Prabhjot" persona)
    
✅ All responses received successfully.
```

---

## ✅ Features

- Sequence-aware conversational memory using `correlation_id`
- Python backend using `pb_utils` for prompt processing
- Streaming-style responses based on accumulated prompt history
- Simulated persona response (e.g., "Prabhjot")

---

## 🛠️ Notes

- Ensure your model directory is properly structured:
  ```
  /models/
    llama1/
      1/
        model.py
        ...
      config.pbtxt
  ```

- `correlation_id` is used to track conversational context across requests.
- Flags like `TRITONSERVER_REQUEST_FLAG_SEQUENCE_START` and `SEQUENCE_END` help maintain session history.

---

## 📂 File Overview

| File              | Purpose                             |
|-------------------|-------------------------------------|
| `model.py`        | Custom Python backend model logic   |
| `config.pbtxt`    | Triton model configuration          |
| `test_llama.py`   | Python client script to test model  |
| `README.md`       | This documentation                  |

---

## 🧠 Example Use Cases

- Conversational agents
- Persona-based storytelling
- Sequential prompt processing with contextual memory

---

## 📧 Contact

For help, reach out to the maintainers or open an issue in your repo.