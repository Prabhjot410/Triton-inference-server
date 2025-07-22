import os

os.environ["TRANSFORMERS_CACHE"] = "/opt/tritonserver/model_repository/llama1/hf-cache"

import json
import numpy as np
import torch
import transformers
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args["model_config"])
        self.model_params = self.model_config.get("parameters", {})

        default_hf_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        default_max_gen_length = "512"

        hf_model = self.model_params.get("huggingface_model", {}).get(
            "string_value", default_hf_model
        )

        self.max_output_length = int(
            self.model_params.get("max_output_length", {}).get(
                "string_value", default_max_gen_length
            )
        )

        self.logger.log_info(f"Max output length: {self.max_output_length}")
        self.logger.log_info(f"Loading HuggingFace model: {hf_model}...")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            hf_model
        )

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=hf_model,
            torch_dtype=torch.float16,
            tokenizer=self.tokenizer,
            device_map="auto",
        )

        # Maintain session history using sequence_id
        self.session_history = {}

    def execute(self, requests):
        responses = []
        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
            prompt = input_tensor.as_numpy()[0].decode("utf-8")

            # Correct way to get sequence_id and flags
            sequence_id = request.correlation_id()
            flags = request.flags()
            is_start = flags & pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_START
            is_end = flags & pb_utils.TRITONSERVER_REQUEST_FLAG_SEQUENCE_END

            # Initialize session history dict if not already
            if not hasattr(self, "session_history"):
                self.session_history = {}

            # Manage session prompt history
            if is_start:
                self.session_history[sequence_id] = prompt
            else:
                self.session_history[sequence_id] += f"\n{prompt}"

            full_prompt = self.session_history[sequence_id]

            # Generate response
            response = self.generate(full_prompt)

            # Clean up session
            if is_end and sequence_id in self.session_history:
                del self.session_history[sequence_id]

            responses.append(response)

        return responses


    def generate(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_token_count = input_ids.shape[1]

        sequences = self.pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=self.max_output_length,
        )

        output_tensors = []
        texts = []
        generated_token_counts = []

        for i, seq in enumerate(sequences):
            generated_text = seq["generated_text"]
            self.logger.log_info(f"Sequence {i+1}: {generated_text}")

            total_token_count = len(self.tokenizer.encode(generated_text))
            generated_count = total_token_count - input_token_count
            self.logger.log_info(f"Generated tokens: {generated_count}")

            texts.append(generated_text)
            generated_token_counts.append(generated_count)

        tensor_text = pb_utils.Tensor("text_output", np.array(texts, dtype=np.object_))
        tensor_token_count = pb_utils.Tensor("token_count", np.array(generated_token_counts, dtype=np.int32))

        output_tensors.extend([tensor_text, tensor_token_count])
        return pb_utils.InferenceResponse(output_tensors=output_tensors)

    def finalize(self):
        print("Cleaning up...")
