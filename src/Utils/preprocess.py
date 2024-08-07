from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision 
from Utils.config import parse_args


args = parse_args()

transformer = transforms.Compose([
    transforms.Resize((args.im_s,args.im_s)),
    transforms.RandomResizedCrop(args.im_s),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# train_path = 'data/train'
# test_path = 'data/test'

train_path = args.train_data
test_path = args.test_data

train_loader = DataLoader(
    torchvision.datasets.ImageFolder(train_path , transform = transformer),
    batch_size = args.bs , shuffle = True
)

test_loader = DataLoader(
    torchvision.datasets.ImageFolder(test_path , transform = transformer),
    batch_size = args.bs, shuffle = True
)