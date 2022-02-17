from matplotlib.cbook import flatten
import numpy as np
import cv2 as cv
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def image(path: str, encoding='srgb', gamma=2.2):
    if encoding not in ['srgb', 'gamma', 'linear']:
        raise RuntimeError("encoding must be one of 'srgb','gamma','linear'")
    ext = path.split('.')[-1]

    img = cv.imread(path, cv.IMREAD_COLOR, dtype=np.float32)
    # if ext == 'exr':
    #     img = img.astype(np.float32)
    # elif ext in ['png','jpg','jpeg']:
    #     img = img.astype(np.float32) / 255.0
    # else:
    #     raise RuntimeError('unsupported format')
    if encoding == 'gamma':
        img = img ** (gamma)
    elif encoding == 'srgb':
        mask = img <= 0.04045
        img = np.select([mask, ~mask], [img / 12.92,
                        ((img + 0.055) / 1.055) ** 2.4])
    return torch.tensor(img).to(device).permute(2, 0, 1)


PERMUTATION = [151, 160, 137, 91, 90, 15,
               131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23,
               190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33,
               88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166,
               77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244,
               102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196,
               135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123,
               5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42,
               223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
               129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228,
               251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107,
               49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254,
               138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180]
PERMUTATION = torch.tensor(PERMUTATION + PERMUTATION,
                           dtype=torch.int).to(device)


def lerp(a, x, y):
    return a * y + (1.0 - a) * x


def _perlin3d(x, y, z):
    """
    https://cs.nyu.edu/~perlin/noise/
    """
    def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)

    def grad(hash, x, y, z):
        h = hash & 15
        u = torch.where(h < 8, x, y)
        v = torch.where(h < 4, y, torch.where((h == 12) | (h == 14), x, z))
        return torch.where((h & 1) == 0, u, -u) + torch.where((h & 2) == 0, v, -v)
    X = (torch.floor(x).int() & 255).long()
    Y = (torch.floor(y).int() & 255).long()
    Z = (torch.floor(z).int() & 255).long()

    x -= torch.floor(x)
    y -= torch.floor(y)
    z -= torch.floor(z)

    u, v, w = fade(x), fade(y), fade(z)

    p = PERMUTATION

    A = p[X]+Y
    AA = p[A]+Z
    AB = p[A+1]+Z
    B = p[X+1]+Y
    BA = p[B]+Z
    BB = p[B+1]+Z

    return lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z),
                                grad(p[BA], x-1, y, z)),
                        lerp(u, grad(p[AB], x, y-1, z),
                             grad(p[BB], x-1, y-1, z))),
                lerp(v, lerp(u, grad(p[AA+1], x, y, z-1),
                             grad(p[BA+1], x-1, y, z-1)),
                     lerp(u, grad(p[AB+1], x, y-1, z-1),
                          grad(p[BB+1], x-1, y-1, z-1))))


def perlin(dim, scale=(1, 1), offset=(0, 0)):
    xs = torch.arange(dim[0]).to(device)
    ys = torch.arange(dim[1]).to(device)
    grid_x, grid_y = torch.meshgrid(xs, ys)
    grid_x = torch.flatten(grid_x.to(device))
    grid_y = torch.flatten(grid_y.to(device))
    xs = grid_x.float()
    ys = grid_y.float()
    xs /= dim[0]
    ys /= dim[1]
    xs += offset[0]
    ys += offset[1]
    xs *= scale[0] * 4
    ys *= scale[1] * 4
    noise = _perlin3d(xs, ys, torch.zeros((1,)).to(device)).to(device)
    img = torch.ones((1, dim[0], dim[1])).to(device)
    img[0, grid_x, grid_y] = noise
    return img


def fbm(dim, scale, offset, noise: callable, levels, multiplier):
    img = noise(dim, scale, offset)
    for i in range(1, levels):
        img += 1.0 / (multiplier**i) * noise(dim,
                                             (scale[0] * multiplier, scale[1] * multiplier, offset))
    return img


def write_tensor_exr(path, tensor):
    cv.imwrite(path, tensor.permute(1, 2, 0).detach().cpu().numpy())


def write_tensor_png(path, tensor, gamma=2.2):
    tensor.clamp_(0.0, 1.0)
    cv.imwrite(path, (tensor.permute(
        1, 2, 0).detach().cpu().numpy()**(1.0/gamma)) * 255.0)


def write(path, img):
    ext = path.split('.')[-1]
    if ext == 'exr':
        write_tensor_exr(path, img)
    else:
        write_tensor_png(path, img)


if __name__ == '__main__':
    # p = perlin((512, 512), scale=(1, 1))
    p = fbm((512, 512), (4, 4), (0, 0), perlin, 5, 2.0) * 0.5 + 0.5
    write_tensor_png('out.png', p, 1.0)
