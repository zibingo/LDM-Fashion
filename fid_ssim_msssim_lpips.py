
from pytorch_fid import fid_score
from pytorch_msssim import ssim,ms_ssim
import lpips
import os
from PIL import Image
import json
from tqdm import tqdm 
import threading
import datetime 
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])
results = {}
def compute_fid(real_images_folder,generated_images_folder):
    # 计算FID距离值
    print("computing fid...")
    fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                                    device="cuda",
                                                    batch_size=1,
                                                    dims=2048)
    fid_value = round(fid_value,3)
    print("fid:",fid_value)
    results["fid"] = fid_value
    return fid_value

def compute_lpips(real_images_folder,generated_images_folder):
    print("computing lpips...")
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    real_images_name = os.listdir(real_images_folder)
    lpips_alex_total = 0.0

    for filename in tqdm(real_images_name,desc="LPIPS"):
        img1 = Image.open(os.path.join(real_images_folder, filename))
        img2 = Image.open(os.path.join(generated_images_folder,filename))
        # lpips指标需要归一化到【-1,1】之间
        img1 = (2.0 * transform(img1) - 1.0).unsqueeze(0).cuda()
        img2 = (2.0 * transform(img2) - 1.0).unsqueeze(0).cuda()
        # 计算图像之间的结构相似性
        d1 = loss_fn_alex.forward(img1, img2)
        lpips_alex_total += d1.item()

    lpips_alex_avg = round(lpips_alex_total/len(real_images_name),3)
    print("lpips_alex_avg:",lpips_alex_avg)
    results["lpips_alex_avg"] = lpips_alex_avg
    return lpips_alex_avg

def compute_ssim(real_images_folder,generated_images_folder):
    print("computing ssim...")
    real_images_name = os.listdir(real_images_folder)
    ssim_total = 0.0
    ms_ssim_total = 0.0
    for filename in tqdm(real_images_name,desc="SSIM"):
        img1 = Image.open(os.path.join(real_images_folder, filename))
        img2 = Image.open(os.path.join(generated_images_folder,filename))
        img1 = transform(img1).unsqueeze(0)
        img2 = transform(img2).unsqueeze(0)
        # ssim和ms_ssim指标归一化到【0,1】之间，data_range=1
        ssim_total += ssim(img1, img2, data_range=1, size_average=False).item()
        ms_ssim_total += ms_ssim(img1, img2, data_range=1, size_average=False).item()

    ssim_avg = round(ssim_total/len(real_images_name),3)
    ms_ssim_avg = round(ms_ssim_total/len(real_images_name),3)
    print("ssim_avg:",ssim_avg)
    print("ms_ssim_avg:",ms_ssim_avg)
    results["ssim_avg"] = ssim_avg
    results["ms_ssim_avg"] = ms_ssim_avg
    return ssim_avg,ms_ssim_avg

if __name__ == "__main__":
    real_images_folder = r'mydataset1/test/gt' 
    generated_images_folder = r"gen_data"

    thread1 = threading.Thread(target=compute_fid, args=(real_images_folder,generated_images_folder))
    thread2 = threading.Thread(target=compute_lpips, args=(real_images_folder,generated_images_folder))
    thread3 = threading.Thread(target=compute_ssim, args=(real_images_folder,generated_images_folder))
    thread1.start()
    thread2.start()
    thread3.start()
    thread1.join()
    thread2.join()
    thread3.join()

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    data = {
        "time": now,
        "real_images_folder": real_images_folder,
        "generated_images_folder": generated_images_folder,
        "fid": "{} ↓ ".format(results.get("fid")),
        "lpips_alex_avg": "{} ↓ ".format(results.get("lpips_alex_avg")),
        "ssim_avg": "{} ↑ ".format(results.get("ssim_avg")),
        "ms_ssim_avg": "{} ↑ ".format(results.get("ms_ssim_avg"))
    }
    # 将数据保存到 JSON 文件
    with open('评估结果_{}.json'.format(now), 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=2, ensure_ascii=False)