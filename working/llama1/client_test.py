import sys
import queue
import uuid
import numpy as np
from functools import partial
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


# Holds completed async results
class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback to handle responses
def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


# Send sequence of text prompts
def async_stream_send(triton_client, prompts, sequence_id, model_name):
    for i, prompt in enumerate(prompts, start=1):
        input_tensor = grpcclient.InferInput("text_input", [1], "BYTES")
        input_tensor.set_data_from_numpy(np.array([prompt.encode("utf-8")], dtype=object))

        output_tensor = grpcclient.InferRequestedOutput("text_output")

        triton_client.async_stream_infer(
            model_name=model_name,
            inputs=[input_tensor],
            outputs=[output_tensor],
            request_id=f"{sequence_id}_{i}",
            sequence_id=sequence_id,
            sequence_start=(i == 1),
            sequence_end=(i == len(prompts)),
        )


def main():
    model_name = "llama1"
    model_version = ""
    prompts = ["Hello, my name is Prabhjot", "what is my name?", "Tell me a joke."]  # Change as needed
    sequence_id = 12345  # Must be non-zero

    user_data = UserData()

    try:
        with grpcclient.InferenceServerClient(url="localhost:8001", verbose=False) as triton_client:
            triton_client.start_stream(callback=partial(callback, user_data))

            async_stream_send(
                triton_client=triton_client,
                prompts=prompts,
                sequence_id=sequence_id,
                model_name=model_name
            )

            # Receive all responses
            for i in range(len(prompts)):
                result = user_data._completed_requests.get()

                if isinstance(result, InferenceServerException):
                    print(f"Request {i+1} failed:", result)
                    sys.exit(1)

                output = result.as_numpy("text_output")
                print(f"[{i+1}] Prompt: {prompts[i]}")
                print("    ➜ Response:", output[0].decode("utf-8") if output is not None else "No Output")

            print("✅ All responses received successfully.")

    except Exception as e:
        print("Error occurred:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
