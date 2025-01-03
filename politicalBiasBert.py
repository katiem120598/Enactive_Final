from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

text = """The Department of Energy handles U.S. energy diplomacy, administers the Strategic Petroleum Reserve - which Trump has said he wants to replenish - and runs grant and loan programs to advance energy technologies, such as the Loan Programs Office.
The secretary also oversees the aging U.S. nuclear weapons complex, nuclear energy waste disposal, and 17 national labs.
If confirmed by the Senate, Wright will replace Jennifer Granholm, a supporter of electric vehicles, emerging energy sources like geothermal power and a backer of carbon-free wind, solar and nuclear energy.
 """

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")


inputs = tokenizer(text, return_tensors="pt")
labels = torch.tensor([0])
outputs = model(**inputs, labels=labels)
loss, logits = outputs[:2]

# [0] -> left 
# [1] -> center
# [2] -> right
print(logits.softmax(dim=-1)[0].tolist()) 