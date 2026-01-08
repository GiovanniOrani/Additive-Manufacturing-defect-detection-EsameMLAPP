import os
import glob
import torch
from ConSinGAN.config import get_arguments
import ConSinGAN.functions as functions
from ConSinGAN.imresize import imresize_to_shape
from tqdm import tqdm

def make_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def generate_samples(netG, reals_shapes, noise_amp, scale_w=1.0, scale_h=1.0, reconstruct=False, n=50):
    if reconstruct:
        reconstruction = netG(fixed_noise, reals_shapes, noise_amp)
        if opt.train_mode == "generation" or opt.train_mode == "retarget":
            functions.save_image('{}/reconstruction.jpg'.format(dir2save), reconstruction.detach())
            functions.save_image('{}/real_image.jpg'.format(dir2save), reals[-1].detach())
        elif opt.train_mode == "harmonization" or opt.train_mode == "editing":
            functions.save_image('{}/{}_wo_mask.jpg'.format(dir2save, _name), reconstruction.detach())
            functions.save_image('{}/real_image.jpg'.format(dir2save), imresize_to_shape(real, reals_shapes[-1][2:], opt).detach())
        return reconstruction

    if scale_w == 1. and scale_h == 1.:
        dir2save_parent = os.path.join(dir2save, "random_samples")
    else:
        reals_shapes = [[r_shape[0], r_shape[1], int(r_shape[2]*scale_h), int(r_shape[3]*scale_w)] for r_shape in reals_shapes]
        dir2save_parent = os.path.join(dir2save, "random_samples_scale_h_{}_scale_w_{}".format(scale_h, scale_w))

    make_dir(dir2save_parent)

    for idx in range(n):
        noise = functions.sample_random_noise(opt.train_stages - 1, reals_shapes, opt)
        sample = netG(noise, reals_shapes, noise_amp)
        functions.save_image('{}/gen_sample_{}.jpg'.format(dir2save_parent, idx), sample.detach())


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--model_dir', help='input image name', required=True)
    parser.add_argument('--gpu', type=int, help='which GPU', default=0)
    parser.add_argument('--num_samples', type=int, help='which GPU', default=50)
    parser.add_argument('--naive_img', help='naive input image  (harmonization or editing)', default="")
    parser.add_argument('--naive_dir', help='naive input directory  (harmonization or editing)', default="")
    parser.add_argument('--naive_dir_pattern', help='naive input image pattern  (harmonization or editing)', default="")

    opt = parser.parse_args()
    _gpu = opt.gpu
    _naive_img = opt.naive_img
    __model_dir = opt.model_dir
    opt = functions.load_config(opt)
    opt.gpu = _gpu
    opt.naive_img = _naive_img
    opt.model_dir = __model_dir

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
        opt.device = "cuda:{}".format(opt.gpu)

    dir2save = os.path.join(opt.model_dir, "Evaluation")
    make_dir(dir2save)

    print("Loading models...")
    netG = torch.load('%s/G.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()), weights_only=False)
    fixed_noise = torch.load('%s/fixed_noise.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    reals = torch.load('%s/reals.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    noise_amp = torch.load('%s/noise_amp.pth' % opt.model_dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    reals_shapes = [r.shape for r in reals]

    
    if opt.naive_dir!="":
        # Define pattern to search all PNG files
        pattern = os.path.join(opt.naive_dir, f"*{opt.naive_dir_pattern}*")
        image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")

        # Find all files that match the pattern.
        images = glob.glob(pattern)

        # Filter the images that do not contain 'mask' in the name.
        naive_images = [
            img for img in images
            if 'mask' not in os.path.basename(img).lower()
            and "harmonized" not in os.path.basename(img).lower()
            and os.path.basename(img).lower().endswith(image_extensions)
        ]
        pbar = tqdm(total=len(naive_images), desc=opt.train_mode)
    else:
        pbar = None
        naive_images = [opt.naive_img]

    index_name=1

    for naive_image in naive_images:
        if opt.train_mode == "generation" or opt.train_mode == "retarget":

            print("Generating Samples...")
            with torch.no_grad():
                # Generate reconstruction
                generate_samples(netG, reals_shapes, noise_amp, reconstruct=True)

                # Generate random samples of normal resolution
                rs0 = generate_samples(netG, reals_shapes, noise_amp, n=opt.num_samples)

                # Generate random samples of different resolution
                generate_samples(netG, reals_shapes, noise_amp, scale_w=2, scale_h=1, n=opt.num_samples)
                generate_samples(netG, reals_shapes, noise_amp, scale_w=1, scale_h=2, n=opt.num_samples)
                generate_samples(netG, reals_shapes, noise_amp, scale_w=2, scale_h=2, n=opt.num_samples)

        elif opt.train_mode == "harmonization" or opt.train_mode == "editing":
            opt.noise_scaling = 0.1
            _name = "harmonized" if opt.train_mode == "harmonization" else "edited"
            _filename, _extension = os.path.splitext(naive_image)
            _filename = os.path.basename(_filename)
            real = functions.read_image_dir(naive_image, opt)
            real = imresize_to_shape(real, reals_shapes[0][2:], opt)
            fixed_noise[0] = real
            if opt.train_mode == "editing":
                fixed_noise[0] = fixed_noise[0] + opt.noise_scaling * \
                                                functions.generate_noise([opt.nc_im, fixed_noise[0].shape[2],
                                                                            fixed_noise[0].shape[3]],
                                                                            device=opt.device)

            out = generate_samples(netG, reals_shapes, noise_amp, reconstruct=True)
            mask_file_name = '{}_mask{}'.format(naive_image[:-4], naive_image[-4:])

            if os.path.exists(mask_file_name):
                mask = functions.read_image_dir(mask_file_name, opt)

                if mask.shape[3] != out.shape[3]:
                    mask = imresize_to_shape(mask, [out.shape[2], out.shape[3]], opt)

                mask = functions.dilate_mask(mask, opt)
                out = (1 - mask) * reals[-1] + mask * out
                
                functions.save_image('{}/{}_{}{}'.format(dir2save, _filename,_name,_extension), out.detach())
            else:
                print("Warning: mask {} not found.".format(mask_file_name))
                print("Harmonization/Editing only performed without mask.")

        elif opt.train_mode == "animation":
            print("Generating GIFs...")
            for _start_scale in range(3):
                for _beta in range(80, 100, 5):
                    functions.generate_gif(dir2save, netG, fixed_noise, reals, noise_amp, opt,
                                        alpha=0.1, beta=_beta/100.0, start_scale=_start_scale, num_images=100, fps=10)
        
        if 'pbar' in globals():
            pbar.update(1)

    if pbar:
        pbar.close()
    print("Done. Results saved at: {}".format(dir2save))

