# People vs Food -- Binary Image Classifier

A PyTorch CNN that classifies images as either **people** or **food**.

## Requirements

- Python 3
- PyTorch
- torchvision
- matplotlib

Install dependencies:

```
pip install torch torchvision matplotlib
```

## Project Structure

```
data/
  train/people/    (25 images)
  train/food/      (25 images)
  val/people/      (2 images)
  val/food/        (2 images)
  test/people/     (5 images)
  test/food/       (5 images)
train.py             Training and evaluation script
final_report.txt     Project report with results
outputs/
  model.pth          Trained model weights
  training_loss.png  Training loss plot
  test_accuracy.png  Test accuracy plot
  training_log.txt   Per-epoch log
```

## Usage

### Train the model

```
python3 train.py
```

This runs the full pipeline: loads the dataset, trains the CNN for 30 epochs, evaluates on the validation and test sets each epoch, then saves the model and output files to `outputs/`.

### Load the trained model

```python
import torch
from train import PeopleFoodCNN

model = PeopleFoodCNN()
model.load_state_dict(torch.load("outputs/model.pth", map_location="cpu"))
model.eval()
```

### Classify a single image

```python
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

img = transform(Image.open("path/to/image.jpg").convert("RGB")).unsqueeze(0)
output = model(img)
prediction = output.argmax(dim=1).item()
label = "food" if prediction == 0 else "people"
print(label)
```

## Results

- **Final test accuracy:** 80.00%
- **Best test accuracy:** 100.00% (epoch 3)

See `final_report.txt` for the full writeup and `outputs/training_log.txt` for per-epoch metrics.
