import base64
import json
import time
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput, InferRequestedOutput
from tritonclient.utils import InferenceServerException

TRITON_SERVER_URL = "localhost:8001"
MODEL_NAME = "llama1"
MODEL_VERSION = ""
SEQUENCE_ID = 12345  # Must be non-zero

def interactive_sequence(triton_client, sequence_id):
    print("Start entering your prompts. Type 'END' to finish the sequence.\n")

    step = 0
    started = False

    while True:
        prompt = input(f"[Prompt {step + 1}]: ")
        if prompt.strip().upper() == "END":
            if not started:
                print("❌ Cannot send END without a valid START.")
                return
            print("\nEnding the sequence...")
            break

        # Prepare input tensor
        input_tensor = grpcclient.InferInput("text_input", [1], "BYTES")
        input_tensor.set_data_from_numpy(np.array([prompt.encode("utf-8")], dtype="object"))

        outputs = [
            grpcclient.InferRequestedOutput("text_output"),
            grpcclient.InferRequestedOutput("token_count")
        ]

        # Only set START on the first valid prompt
        is_start = (step == 0)
        try:
            response = triton_client.infer(
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                inputs=[input_tensor],
                outputs=outputs,
                sequence_id=sequence_id,
                sequence_start=is_start,
                sequence_end=False
            )

            if is_start:
                started = True  # Mark that sequence has started

            generated_text = response.as_numpy("text_output")[0].decode("utf-8")
            token_count = response.as_numpy("token_count")[0]
            print(f"\n[Generated Response {step + 1}]:\n{generated_text}")
            print(f"Generated Tokens: {token_count}\n")

            step += 1

        except InferenceServerException as e:
            print(f"Error during inference: {str(e)}")
            return

    # Send dummy prompt to end session
    try:
        final_input_tensor = grpcclient.InferInput("text_input", [1], "BYTES")
        final_input_tensor.set_data_from_numpy(np.array(["<|end|>".encode("utf-8")], dtype="object"))

        triton_client.infer(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            inputs=[final_input_tensor],
            outputs=[
                grpcclient.InferRequestedOutput("text_output"),
                grpcclient.InferRequestedOutput("token_count")
            ],
            sequence_id=sequence_id,
            sequence_start=False,
            sequence_end=True
        )
        print("✅ Session ended cleanly.")
    except Exception as e:
        print(f"Failed to send sequence_end: {str(e)}")



if __name__ == "__main__":
    with grpcclient.InferenceServerClient(url=TRITON_SERVER_URL, verbose=False) as client:
        print(f"🧠 Connected to Triton Server at {TRITON_SERVER_URL}")
        interactive_sequence(client, SEQUENCE_ID)
