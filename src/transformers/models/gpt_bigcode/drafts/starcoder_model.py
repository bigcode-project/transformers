from transformers import AutoModelForCausalLM


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("bigcode/starcoderbase-1b")
    states = model.state_dict()
    print(states.keys())
