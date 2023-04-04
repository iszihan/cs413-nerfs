""" axis transformation, rotation, ..., any other utils"""
import torch 
import inspect
# torch utils 
def printarr(*arrs, data=True, short=True, max_width=100):

    # helper for below
    def compress_str(s):
        return s.replace('\n', '')
    name_align = ">" if short else "<"

    # get the name of the tensor as a string
    frame = inspect.currentframe().f_back
    try:
        # first compute some length stats
        name_len = -1
        dtype_len = -1
        shape_len = -1
        default_name = "[unnamed]"
        for a in arrs:
            name = default_name
            for k, v in frame.f_locals.items():
                if v is a:
                    name = k
                    break
            name_len = max(name_len, len(name))
            dtype_len = max(dtype_len, len(str(a.dtype)))
            shape_len = max(shape_len, len(str(a.shape)))
        len_left = max_width - name_len - dtype_len - shape_len - 5

        # now print the acual arrays
        for a in arrs:
            name = default_name
            for k, v in frame.f_locals.items():
                if v is a:
                    name = k
                    break
            print(
                f"{name:{name_align}{name_len}} {str(a.dtype):<{dtype_len}} {str(a.shape):>{shape_len}}", end='')
            if data:
                # print the contents of the array
                print(": ", end='')
                flat_str = compress_str(str(a))
                if len(flat_str) < len_left:
                    # short arrays are easy to print
                    print(flat_str)
                else:
                    # long arrays
                    if short:
                        # print a shortented version that fits on one line
                        if len(flat_str) > len_left - 4:
                            flat_str = flat_str[:(len_left-4)] + " ..."
                        print(flat_str)
                    else:
                        # print the full array on a new line
                        print("")
                        print(a)
                print(a.isnan().any())
                print(a.isinf().any())
            else:
                print("")  # newline
    finally:
        del frame


def writable_image(img):
    img_min = torch.min(img)
    img_max = torch.max(img)
    img = (img - img_min) * (255 / (img_max - img_min))
    #img = (img-lo) * (255 / (hi-lo))
    img = torch.round(img).clip(0,255).to(torch.uint8)
    return img

def toNP(x):
    return x.detach().to(torch.device('cpu')).numpy()

def spherical_poses():
    return None 
