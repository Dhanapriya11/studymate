import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import logging

# Suppress transformers warnings
logging.set_verbosity_error()

class HFLLM:
    def __init__(self, model_id=None, device="cpu", hf_token=None):
        if model_id is None:
            model_id = "google/flan-t5-small"
        
        self.model_id = model_id
        self.device = device
        self.hf_token = hf_token
        
        model_kwargs = {}
        if hf_token:
            model_kwargs["token"] = hf_token
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, **model_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            device_index = -1 if device == "cpu" else 0
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_index,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                num_beams=1,
                early_stopping=False
            )
        except Exception as e:
            if "does not appear to have a file named pytorch_model.bin" in str(e):
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_id, **model_kwargs)
                    self.model = AutoModelForCausalLM.from_pretrained(model_id, from_tf=True, **model_kwargs)
                    device_index = -1 if device == "cpu" else 0
                    self.pipe = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        device=device_index,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=0.7,
                        num_beams=1,
                        early_stopping=False
                    )
                except Exception:
                    self._load_fallback_model()
            else:
                self._load_fallback_model()

    def _load_fallback_model(self):
        fallback_model = "sshleifer/tiny-gpt2"
        print(f"Warning: Failed to load primary model, falling back to {fallback_model}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForCausalLM.from_pretrained(fallback_model)
            device_index = -1 if self.device == "cpu" else 0
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device_index,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                num_beams=1,
                early_stopping=False
            )
        except Exception as fallback_e:
            print(f"Warning: Failed to load fallback model {fallback_model}: {fallback_e}")
            self._create_dummy_model()

    def _create_dummy_model(self):
        self.tokenizer = None
        self.model = None
        self.pipe = None
        self.is_dummy = True

    def generate(self, prompt: str, max_new_tokens=None) -> str:
        if max_new_tokens is None:
            max_new_tokens = 150
            
        if hasattr(self, 'is_dummy') and self.is_dummy:
            return ("I'm sorry, but I'm currently unable to generate responses "
                    "due to model loading issues. Please check your internet connection "
                    "and disk space, or try a different model.")
            
        try:
            output = self.pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                num_beams=1,
                early_stopping=False
            )
            
            # Extract text depending on output format
            if isinstance(output, list) and len(output) > 0:
                if isinstance(output[0], dict) and "generated_text" in output[0]:
                    result = output[0]["generated_text"]
                elif isinstance(output[0], str):
                    result = output[0]
                else:
                    result = str(output[0])
            elif isinstance(output, str):
                result = output
            else:
                return "Unable to generate response - unexpected output format"

            # Remove echoed prompt if present
            if prompt in result:
                result = result.split(prompt)[-1].strip()

            # Keep only bullet point lines
            bullet_lines = [
                line.strip() for line in result.split("\n")
                if line.strip().startswith(("•", "-", "*"))
            ]
            return "\n".join(bullet_lines) if bullet_lines else result.strip()

        except Exception as e:
            return (f"Error generating response: {str(e)}. "
                    "Consider changing the download location by setting the "
                    "HF_HOME environment variable to a drive with more space.")
