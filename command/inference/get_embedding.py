from command import configs
from fairseq.models.trex import TrexModel
import torch
import numpy as np

trex = TrexModel.from_pretrained(f'checkpoints/similarity',
                                 checkpoint_file='checkpoint_best.pt',
                                 data_name_or_path=f'data-bin/similarity')

trex.eval()

samples0 = {field: [] for field in configs.fields}
samples1 = {field: [] for field in configs.fields}
labels = []

for field in configs.fields:
    with open(f'data-src/similarity/valid.{field}.input0', 'r') as f:
        for line in f:
            samples0[field].append(line.strip())
for field in configs.fields:
    with open(f'data-src/similarity/valid.{field}.input1', 'r') as f:
        for line in f:
            samples1[field].append(line.strip())
with open(f'data-src/similarity/valid.label', 'r') as f:
    for line in f:
        labels.append(float(line.strip()))

top = 1582
similarities = []
# emb = []

# f_emb0 = open("data/emb0.txt", 'a')
# f_emb1 = open("data/emb1.txt", 'a')
f_similarity = open("data/similarity.txt", 'a')
# f_result = open("data/result.txt", 'a')

for sample_idx in range(top):
    sample0 = {field: samples0[field][sample_idx] for field in configs.fields}
    sample1 = {field: samples1[field][sample_idx] for field in configs.fields}
    label = labels[sample_idx]

    sample0_tokens = trex.encode(sample0)
    sample1_tokens = trex.encode(sample1)

    emb0 = trex.predict('similarity', sample0_tokens)
    emb1 = trex.predict('similarity', sample1_tokens)
    # # emb.append(emb1)
    # # print (emb1)
    # # print(emb0, emb1)
    # # f_emb0.write(str(emb0) + '\n')
    # list_emb1 = emb1.detach().numpy().tolist()
    # # print (list_emb1)
    # f_emb1.write(str(list_emb1)+'\n')
    similarities.append(torch.cosine_similarity(emb0, emb1)[0].item())
    # f_similarity.write(str(emb1) + '\n')


# print(emb)
pred = np.array(similarities)
result = list(pred)


np.set_printoptions(threshold=np.inf)
# f_result.write(str(result))


f_similarity.write(str(result))
# print(pred)
print(result)

