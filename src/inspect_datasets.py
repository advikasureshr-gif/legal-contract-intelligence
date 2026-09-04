from datasets import load_dataset

dataset = load_dataset(
    "coastalcph/lex_glue",
    "unfair_tos"
)

dataset.save_to_disk("Datasets/unfair_tos")

print("UNFAIR-ToS saved successfully!")
print(dataset)