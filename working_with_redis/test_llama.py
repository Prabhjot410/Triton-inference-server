import random
import time
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput, InferRequestedOutput
from tritonclient.utils import InferenceServerException

TRITON_SERVER_URL = "localhost:8001"
MODEL_NAME = "llama1"
MODEL_VERSION = ""

def generate_random_id():
    # Generate a random 64-bit integer
    return random.randint(1_000_000_000, 9_999_999_999)

def interactive_sequence(triton_client, user_id, sequence_id):
    print(f"\n🧑‍💻 Chat started (user_id={user_id}, session_id={sequence_id})")
    print("Type 'END' to finish the chat.\n")

    step = 0
    started = False

    while True:
        prompt = input(f"[User {user_id} | Prompt {step + 1}]: ").strip()
        if prompt.upper() == "END":
            if not started:
                print("❌ Cannot send END without a valid START.")
                return
            print(f"\n📴 Ending session for user {user_id}...\n")
            break

        user_tensor = grpcclient.InferInput("user_id", [1], "BYTES")
        user_tensor.set_data_from_numpy(np.array([str(user_id).encode("utf-8")], dtype="object"))

        prompt_tensor = grpcclient.InferInput("text_input", [1], "BYTES")
        prompt_tensor.set_data_from_numpy(np.array([prompt.encode("utf-8")], dtype="object"))

        outputs = [
            grpcclient.InferRequestedOutput("text_output"),
            grpcclient.InferRequestedOutput("token_count"),
        ]

        try:
            response = triton_client.infer(
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                inputs=[user_tensor, prompt_tensor],
                outputs=outputs,
                sequence_id=sequence_id,
                sequence_start=(step == 0),
                sequence_end=False,
            )

            started = True
            generated_text = response.as_numpy("text_output")[0].decode("utf-8")
            token_count = response.as_numpy("token_count")[0]
            print(f"\n🤖 [Response]: {generated_text}")
            print(f"🧮 Tokens Generated: {token_count}\n")

            step += 1

        except InferenceServerException as e:
            print(f"❌ Inference failed: {str(e)}")
            return

    # Final END request
    try:
        user_tensor = grpcclient.InferInput("user_id", [1], "BYTES")
        user_tensor.set_data_from_numpy(np.array([str(user_id).encode("utf-8")], dtype="object"))

        end_tensor = grpcclient.InferInput("text_input", [1], "BYTES")
        end_tensor.set_data_from_numpy(np.array(["<|end|>".encode("utf-8")], dtype="object"))

        triton_client.infer(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            inputs=[user_tensor, end_tensor],
            outputs=[
                grpcclient.InferRequestedOutput("text_output"),
                grpcclient.InferRequestedOutput("token_count"),
            ],
            sequence_id=sequence_id,
            sequence_start=False,
            sequence_end=True,
        )
        print("✅ Sequence ended successfully.")
    except Exception as e:
        print(f"⚠️ Final sequence_end request failed: {str(e)}")

if __name__ == "__main__":
    with grpcclient.InferenceServerClient(url=TRITON_SERVER_URL, verbose=False) as client:
        print(f"🔗 Connected to Triton Server at {TRITON_SERVER_URL}")

        # Create random user_id and sequence_id (both integers)
        user_id = generate_random_id()
        sequence_id = generate_random_id()

        interactive_sequence(client, user_id, sequence_id)


