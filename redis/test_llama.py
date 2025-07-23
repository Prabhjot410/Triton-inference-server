import time
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput

TRITON_SERVER_URL = "localhost:8001"
MODEL_NAME = "llama1"
MODEL_VERSION = ""
SEQUENCE_ID = int(time.time())  # Unique sequence
USER_ID = f"user_{SEQUENCE_ID}"

def interactive_sequence(triton_client, sequence_id, user_id):
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

        # Input tensors
        user_tensor = grpcclient.InferInput("user_id", [1], "BYTES")
        user_tensor.set_data_from_numpy(np.array([user_id.encode("utf-8")], dtype="object"))

        prompt_tensor = grpcclient.InferInput("text_input", [1], "BYTES")
        prompt_tensor.set_data_from_numpy(np.array([prompt.encode("utf-8")], dtype="object"))

        inputs = [user_tensor, prompt_tensor]

        outputs = [
            grpcclient.InferRequestedOutput("text_output")
        ]

        is_start = (step == 0)

        try:
            response = triton_client.infer(
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                inputs=inputs,
                outputs=outputs,
                sequence_id=sequence_id,
                sequence_start=is_start,
                sequence_end=False,
            )

            if is_start:
                started = True

            result = response.as_numpy("text_output")[0].decode("utf-8")
            print(f"\n[Generated Response {step + 1}]:\n{result}\n")

            step += 1

        except Exception as e:
            print("❌ Error during inference:", e)
            return

    # END sequence with dummy input
    try:
        end_tensor = grpcclient.InferInput("text_input", [1], "BYTES")
        end_tensor.set_data_from_numpy(np.array(["<|end|>".encode("utf-8")], dtype="object"))

        user_tensor = grpcclient.InferInput("user_id", [1], "BYTES")
        user_tensor.set_data_from_numpy(np.array([user_id.encode("utf-8")], dtype="object"))

        triton_client.infer(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            inputs=[user_tensor, end_tensor],
            outputs=[grpcclient.InferRequestedOutput("text_output")],
            sequence_id=sequence_id,
            sequence_start=False,
            sequence_end=True
        )
        print("✅ Session ended cleanly.")
    except Exception as e:
        print(f"Failed to send sequence_end: {str(e)}")


if __name__ == "__main__":
    with grpcclient.InferenceServerClient(url=TRITON_SERVER_URL) as client:
        print(f"🧠 Connected to Triton Server at {TRITON_SERVER_URL}")
        interactive_sequence(client, SEQUENCE_ID, USER_ID)
