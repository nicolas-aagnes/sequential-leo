from datasets import Human36M
import time
import torch


dataset = Human36M(1, 10)
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=2, num_workers=2)

i = 0
start = time.time()
for batch in dataset:
    x_support, y_support, x_query, y_query = batch
    print(
        "Got",
        x_support.shape,
        y_support.shape,
        x_query.shape,
        y_query.shape,
        "in",
        time.time() - start,
    )
    start = time.time()
    i += 1
    if i == 10:
        break
