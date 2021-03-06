import cv2
import numpy as np
import datetime
import os 
from PIL import Image


def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.

    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))

class DarkChannelPrior(object):
    def __init__(self, w=0.95, mf_r=7, gf_r=81, gf_eps=0.001, maxV1=0.80, bGamma=False):
        self.w = w  # 去雾程度
        self.mf_r = mf_r
        self.gf_r = gf_r
        self.gf_eps = gf_eps
        self.maxV1 = maxV1
        self.bGamma = bGamma

    def zmMinFilterGray(self, src, r=7):
        '''最小值滤波，r是滤波器半径'''
        return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))

    def guidedfilter(self, I, p, r, eps):
        height, width = I.shape
        m_I = cv2.boxFilter(I, -1, (r, r))
        m_p = cv2.boxFilter(p, -1, (r, r))
        m_Ip = cv2.boxFilter(I * p, -1, (r, r))
        cov_Ip = m_Ip - m_I * m_p

        m_II = cv2.boxFilter(I * I, -1, (r, r))
        var_I = m_II - m_I * m_I

        a = cov_Ip / (var_I + eps)
        b = m_p - a * m_I

        m_a = cv2.boxFilter(a, -1, (r, r))
        m_b = cv2.boxFilter(b, -1, (r, r))
        return m_a * I + m_b

    def Defog(self, m, r, eps, w, maxV1):                 # 输入rgb图像，值范围[0,1]
        '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
        V1 = np.min(m, 2)                           # 得到暗通道图像
        Dark_Channel = self.zmMinFilterGray(V1, 7)
        # cv2.imshow('20190708_Dark',Dark_Channel)    # 查看暗通道
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        V1 = self.guidedfilter(V1, Dark_Channel, r, eps)  # 使用引导滤波优化
        bins = 2000
        ht = np.histogram(V1, bins)                  # 计算大气光照A
        d = np.cumsum(ht[0]) / float(V1.size)
        for lmax in range(bins - 1, 0, -1):
            if d[lmax] <= 0.999:
                break
        A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
        V1 = np.minimum(V1 * w, maxV1)               # 对值范围进行限制
        return V1, A

    def deHaze(self, m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
        m = m/255.0  # 归一化
        Y = np.zeros(m.shape)
        Mask_img, A = self.Defog(m, r, eps, w, maxV1)             # 得到遮罩图像和大气光照

        for k in range(3):
            Y[:, :, k] = (m[:, :, k] - Mask_img)/(1-Mask_img/A)  # 颜色校正
        Y = np.clip(Y, 0, 1)
        if bGamma:
            Y = Y ** (np.log(0.5) / np.log(Y.mean()))       # gamma校正,默认不进行该操作
        Y = Y*255  # 反归一化
        Y = cv2.convertScaleAbs(Y)
        return Y

# retinex SSR
class SSRetinex(object):
    def __init__(self):
        pass

    def replaceZeroes(self, data):
        min_nonzero = min(data[np.nonzero(data)])
        data[data == 0] = min_nonzero
        return data

    def SSR(self, src_img, size):
        L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
        img = self.replaceZeroes(src_img)
        L_blur = self.replaceZeroes(L_blur)
        dst_Img = cv2.log(img/255.0)
        dst_Lblur = cv2.log(L_blur/255.0)
        dst_IxL = cv2.multiply(dst_Img, dst_Lblur)
        log_R = cv2.subtract(dst_Img, dst_IxL)
        dst_R = cv2.normalize(log_R, None, 0, 255, cv2.NORM_MINMAX)
        log_uint8 = cv2.convertScaleAbs(dst_R)
        return log_uint8

    def SSR_image(self, image):
        size = 3
        b_gray, g_gray, r_gray = cv2.split(image)
        #print('b_shape=', b_gray.shape, 'g_shape=',b_gray.shape, 'r_shape=', b_gray.shape)
        b_gray = self.SSR(b_gray, size)
        g_gray = self.SSR(g_gray, size)
        r_gray = self.SSR(r_gray, size)
        #print('after,b_shape=', b_gray.shape, 'g_shape=',b_gray.shape, 'r_shape=', b_gray.shape)
        result = cv2.merge([b_gray, g_gray, r_gray])
        return result


class Image_Tool(object):
    def __init__(self):
        pass

    def RGB2HSI(self, rgb_img):
        """
        这是将RGB彩色图像转化为HSI图像的函数
        :param rgm_img: RGB彩色图像
        :return: HSI图像
        """
        # 保存原始图像的行列数
        row = np.shape(rgb_img)[0]
        col = np.shape(rgb_img)[1]
        # 对原始图像进行复制
        hsi_img = rgb_img.copy()
        # 对图像进行通道拆分
        B, G, R = cv2.split(rgb_img)
        # 把通道归一化到[0,1]
        [B, G, R] = [i / 255.0 for i in ([B, G, R])]
        H = np.zeros((row, col))  # 定义H通道
        I = (R + G + B) / 3.0  # 计算I通道
        S = np.zeros((row, col))  # 定义S通道
        for i in range(row):
            den = np.sqrt((R[i]-G[i])**2+(R[i]-B[i])*(G[i]-B[i]))
            thetha = np.arccos(0.5*(R[i]-B[i]+R[i]-G[i])/den)  # 计算夹角
            h = np.zeros(col)  # 定义临时数组
            # den>0且G>=B的元素h赋值为thetha
            h[B[i] <= G[i]] = thetha[B[i] <= G[i]]
            # den>0且G<=B的元素h赋值为thetha
            h[G[i] < B[i]] = 2*np.pi-thetha[G[i] < B[i]]
            # den<0的元素h赋值为0
            h[den == 0] = 0
            H[i] = h/(2*np.pi)  # 弧度化后赋值给H通道
        # 计算S通道
        for i in range(row):
            min = []
            # 找出每组RGB值的最小值
            for j in range(col):
                arr = [B[i][j], G[i][j], R[i][j]]
                min.append(np.min(arr))
            min = np.array(min)
            # 计算S通道
            S[i] = 1 - min*3/(R[i]+B[i]+G[i])
            # I为0的值直接赋值0
            S[i][R[i]+B[i]+G[i] == 0] = 0
        # 扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
        hsi_img[:, :, 0] = H*255
        hsi_img[:, :, 1] = S*255
        hsi_img[:, :, 2] = I*255
        return hsi_img

    def HSI2RGB(self, hsi_img):
        """
        这是将HSI图像转化为RGB图像的函数
        :param hsi_img: HSI彩色图像
        :return: RGB图像
        """
        # 保存原始图像的行列数
        row = np.shape(hsi_img)[0]
        col = np.shape(hsi_img)[1]
        # 对原始图像进行复制
        rgb_img = hsi_img.copy()
        # 对图像进行通道拆分
        H, S, I = cv2.split(hsi_img)
        # 把通道归一化到[0,1]
        [H, S, I] = [i / 255.0 for i in ([H, S, I])]
        R, G, B = H, S, I
        for i in range(row):
            h = H[i]*2*np.pi
            # H大于等于0小于120度时
            a1 = h >= 0
            a2 = h < 2*np.pi/3
            a = a1 & a2  # 第一种情况的花式索引
            tmp = np.cos(np.pi / 3 - h)
            b = I[i] * (1 - S[i])
            r = I[i]*(1+S[i]*np.cos(h)/tmp)
            g = 3*I[i]-r-b
            B[i][a] = b[a]
            R[i][a] = r[a]
            G[i][a] = g[a]
            # H大于等于120度小于240度
            a1 = h >= 2*np.pi/3
            a2 = h < 4*np.pi/3
            a = a1 & a2  # 第二种情况的花式索引
            tmp = np.cos(np.pi - h)
            r = I[i] * (1 - S[i])
            g = I[i]*(1+S[i]*np.cos(h-2*np.pi/3)/tmp)
            b = 3 * I[i] - r - g
            R[i][a] = r[a]
            G[i][a] = g[a]
            B[i][a] = b[a]
            # H大于等于240度小于360度
            a1 = h >= 4 * np.pi / 3
            a2 = h < 2 * np.pi
            a = a1 & a2  # 第三种情况的花式索引
            tmp = np.cos(5 * np.pi / 3 - h)
            g = I[i] * (1-S[i])
            b = I[i]*(1+S[i]*np.cos(h-4*np.pi/3)/tmp)
            r = 3 * I[i] - g - b
            B[i][a] = b[a]
            G[i][a] = g[a]
            R[i][a] = r[a]
        rgb_img[:, :, 0] = B*255
        rgb_img[:, :, 1] = G*255
        rgb_img[:, :, 2] = R*255
        return rgb_img

    def clahe(self, image):
        b, g, r = cv2.split(image)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        b = clahe.apply(b)
        g = clahe.apply(g)
        r = clahe.apply(r)
        image_clahe = cv2.merge([b, g, r])
        return image_clahe

    def colorbalance(self, img):
        b, g, r = cv2.split(img)
        B = np.mean(b)
        G = np.mean(g)
        R = np.mean(r)
        K = (R + G + B) / 3
        Kb = K / B
        Kg = K / G
        Kr = K / R
        cv2.addWeighted(b, Kb, 0, 0, 0, b)
        cv2.addWeighted(g, Kg, 0, 0, 0, g)
        cv2.addWeighted(r, Kr, 0, 0, 0, r)
        merged = cv2.merge([b, g, r])
        return merged

    def high_pass_filter(self,img):
        x=cv2.Sobel(img,-1,1,0)
        y=cv2.Sobel(img,-1,0,1)

        absx=cv2.convertScaleAbs(x)
        absy=cv2.convertScaleAbs(y)
        dist=cv2.addWeighted(absx,1,absy,1,0)
        return dist


class Image_Enhance(object):
    def __init__(self):
        pass

    def __call__(self, image):
        img = cv2.imread(image)
        image_tool = Image_Tool()
        starttime = datetime.datetime.now()
        img1 = img.copy()
        img2 = img.copy()
        cb_img=image_tool.colorbalance(img)
        dcp_img = DarkChannelPrior().deHaze(img)  
        #hsi_img = image_tool.RGB2HSI(dcp_img)
        ssr_img = SSRetinex().SSR_image(img)
        #bgr_img = image_tool.HSI2RGB(ssr_img)
        cv2.addWeighted(ssr_img,0.824,dcp_img,0.176,0,img1)
        cv2.addWeighted(img1,0.85,cb_img,0.15,0,img2)
        result = image_tool.clahe(img2)
        endtime = datetime.datetime.now()
        print('image_enhance time consum=%s' %round((endtime-starttime).microseconds/1000000+(endtime-starttime).seconds,6))
        return result
    
    def api(self,img):
        img = img.transpose((1,2,0))
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        image_tool = Image_Tool()
        starttime = datetime.datetime.now()
        img1 = img.copy()
        img2 = img.copy()
        cb_img=image_tool.colorbalance(img)
        dcp_img = DarkChannelPrior().deHaze(img)  
        #hsi_img = image_tool.RGB2HSI(dcp_img)
        ssr_img = SSRetinex().SSR_image(img)
        #bgr_img = image_tool.HSI2RGB(ssr_img)
        cv2.addWeighted(ssr_img,0.5,dcp_img,0.5,0,img1)
        cv2.addWeighted(img1,0.7,cb_img,0.3,0,img2)
        result = image_tool.clahe(img2)
        endtime = datetime.datetime.now()
        print('image_enhance time consum=%s' %round((endtime-starttime).microseconds/1000000+(endtime-starttime).seconds,6))
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        if result.dtype != np.float32:
            result = result.astype(np.float32)
        result = result.transpose((2,0,1))
        return result


if __name__ == '__main__':
    image_enhance = Image_Enhance()
    #img = read_image('/home/lenovo/4T/Taohuang/VOCdevkit/VOC2007/JPEGImages_bak/000802.jpg')
    #result = image_enhance.api(img)
    path = '/home/lenovo/4T/Taohuang/VOCdevkit/VOC2007/JPEGImages'
    jpglist = os.listdir(path)
    for jpg in jpglist:
        result = image_enhance(os.path.join(path, jpg))
        cv2.imwrite(os.path.join(path, jpg), result)
    # result = image_enhance('/home/lenovo/4T/Taohuang/VOCdevkit/VOC2007/JPEGImages_bak/000802.jpg')
    # cv2.imwrite(
    #     '/home/lenovo/4T/Taohuang/simple-faster-rcnn-pytorch/utils/atomization/000001_out.jpg', result)
