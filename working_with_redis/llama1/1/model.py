import os
import json
import numpy as np
import torch
import transformers
import triton_python_backend_utils as pb_utils
import redis

os.environ["TRANSFORMERS_CACHE"] = "/opt/tritonserver/model_repository/llama1/hf-cache"

class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])
        self.model_params = self.model_config.get("parameters", {})

        hf_model = self.model_params.get("huggingface_model", {}).get(
            "string_value", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        self.max_output_length = int(
            self.model_params.get("max_output_length", {}).get("string_value", "50")
        )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=hf_model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.redis_conn = redis.Redis(
            host='redis-17045.c256.us-east-1-2.ec2.redns.redis-cloud.com',
            port=17045,
            decode_responses=True,
            username="default",
            password="rOidZZqUyDMlc8Q3cGMGvhmtWdyVfzjU",
        )

        print("Model and Redis initialized")
        self.session_history = {}

    def execute(self, requests):
        responses = []

        for request in requests:
            try:
                user_id_tensor = pb_utils.get_input_tensor_by_name(request, "user_id")
                user_id = user_id_tensor.as_numpy()[0].decode("utf-8")

                prompt_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
                prompt = prompt_tensor.as_numpy()[0].decode("utf-8")

                sequence_id = request.correlation_id()
                redis_key = f"chat:{user_id}:{sequence_id}"

                flags = request.flags()
                is_start = flags & pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_START
                is_end = flags & pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_END

                print(f"Request received for user: {user_id}, seq_id: {sequence_id}")
                print(f"Prompt: {prompt}")
                print(f"Flags: start={bool(is_start)}, end={bool(is_end)}")

                if is_start:
                    self.session_history[sequence_id] = prompt
                    print(f"New session started for seq_id {sequence_id}")
                else:
                    self.session_history[sequence_id] += f"\n{prompt}"

                full_prompt = self.session_history[sequence_id]
                generated = self.generate(full_prompt, prompt)

                # Store in Redis
                try:
                    history = self.redis_conn.get(redis_key)
                    history = json.loads(history) if history else []
                except Exception as e:
                    print(f"Redis read error: {e}")
                    history = []

                history.append({"role": "user", "content": prompt})
                history.append({"role": "assistant", "content": generated})
                self.redis_conn.set(redis_key, json.dumps(history))

                if is_end and sequence_id in self.session_history:
                    print(f"Ending session for seq_id {sequence_id}")
                    del self.session_history[sequence_id]

                print(f"Response generated: {generated}")

                # output_tensor = pb_utils.Tensor(
                #     "text_output",
                #     np.array([generated.encode("utf-8")], dtype=object)
                # )
                # responses.append(pb_utils.InferenceResponse([output_tensor]))

                # Count the number of tokens in the generated response
                token_count = len(self.tokenizer.encode(generated))

                output_tensor = pb_utils.Tensor(
                    "text_output",
                    np.array([generated.encode("utf-8")], dtype=object)
                )
                token_tensor = pb_utils.Tensor(
                    "token_count",
                    np.array([token_count], dtype=np.int32)
                )

                responses.append(pb_utils.InferenceResponse([output_tensor, token_tensor]))


            except Exception as e:
                error_msg = f"Inference failed: {e}"
                print(error_msg)
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(error_msg)
                ))

        return responses

    def generate(self, full_prompt, latest_user_prompt):
        sequences = self.pipeline(
            full_prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=self.max_output_length,
        )
        full_output = sequences[0]["generated_text"]
        if full_output.startswith(full_prompt):
            return full_output[len(full_prompt):].strip()
        return full_output.strip()

    def finalize(self):
        print("Cleaning up model resources...")
