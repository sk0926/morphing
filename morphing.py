import numpy as np
import cv2
import dlib
from imutils import face_utils
from scipy.spatial import Delaunay

#三角形一つ毎に実行
def morph_points(img1, img2, output, tri1, tri2, tri_dst, alpha) :

    #三角形をちょうど包含する矩形を作成
    rect1 = cv2.boundingRect(tri1)
    rect2 = cv2.boundingRect(tri2)
    rect_dst = cv2.boundingRect(tri_dst)

    #矩形の左上座標を(0, 0)に合わせときの三角形の座標を計算
    tri1_0 = []
    tri2_0 = []
    tri2_dst_0 = []
    for i in range(3):
        tri1_0.append(((tri1[i][0] - rect1[0]),(tri1[i][1] - rect1[1])))
        tri2_0.append(((tri2[i][0] - rect2[0]),(tri2[i][1] - rect2[1])))
        tri2_dst_0.append(((tri_dst[i][0] - rect_dst[0]),(tri_dst[i][1] - rect_dst[1])))

    #変換行列を計算
    mat1 = cv2.getAffineTransform(np.float32(tri1_0), np.float32(tri2_dst_0))
    mat2 = cv2.getAffineTransform(np.float32(tri2_0), np.float32(tri2_dst_0))

    #画像中から矩形部分を取り出す
    img1Rect = img1[rect1[1]:rect1[1]+rect1[3], rect1[0]:rect1[0]+rect1[2]]
    img2Rect = img2[rect2[1]:rect2[1]+rect2[3], rect2[0]:rect2[0]+rect2[2]]

    #アフィン変換を実行
    warped_img1 = cv2.warpAffine(img1Rect, mat1, (rect_dst[2], rect_dst[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    warped_img2 = cv2.warpAffine(img2Rect, mat2, (rect_dst[2], rect_dst[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    #二つの画像を重み付けして足す
    imgRect = warped_img1*(1 - alpha) + warped_img2*alpha

    #三角形のマスクを作成
    mask = np.zeros((rect_dst[3], rect_dst[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri2_dst_0), (1.0, 1.0, 1.0), 16, 0)

    #出力用画像に貼り付け
    output[rect_dst[1]:rect_dst[1]+rect_dst[3], rect_dst[0]:rect_dst[0]+rect_dst[2]] \
        = output[rect_dst[1]:rect_dst[1]+rect_dst[3], rect_dst[0]:rect_dst[0]+rect_dst[2]]*(1-mask) + imgRect*mask


#顔と特徴点の抽出
def detect_landmarks(file_face):

    #顔認識機のインスタンスを生成
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    #認識機で顔認識を実施
    img = cv2.imread(file_face)
    face_img = detector(img, 0)

    size = img.shape

    #検出された顔が1つでない場合はエラー
    if len(face_img) != 1:
        print('Error: cannot detect face or detect some faces', file=sys.stderr)
        sys.exit(1)

    #特徴点を68個抽出
    for i in face_img:
        landmarks = predictor(img, i)
    
    landmarks = face_utils.shape_to_np(landmarks)
    landmarks = landmarks.tolist()

    #画像の四隅と中点を特徴点に追加
    landmarks.append([0, 0])
    landmarks.append([int(size[1]-1), 0])
    landmarks.append([0, int(size[0]-1)])
    landmarks.append([int(size[1]-1), int(size[0]-1)])
    landmarks.append([int(size[1]/2), 0])
    landmarks.append([0, int(size[0]/2)])
    landmarks.append([int(size[1]/2), int(size[0]-1)])
    landmarks.append([int(size[1]-1), int(size[0]/2)])

    #テスト用表示
    for s in landmarks:
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2)
    cv2.imshow("test", img)
    cv2.imwrite('output/kato_img.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #float32にしないとアフィン変換が出来ない
    return np.array(landmarks, dtype='float32')


if __name__ == '__main__' :

    #画像の読み込み
    faceimg1 = 'image/my1.jpg'
    faceimg2 = 'image/my2.jpg'
    img1 = np.float32(cv2.imread(faceimg1))
    img2 = np.float32(cv2.imread(faceimg2))

    for i in range(1):
        if (i % 10) == 0:
            print('progress... {} %'.format(i))
        alpha = i * 0.01
        
        #顔の特徴点を抽出
        points1 = detect_landmarks(faceimg1)
        points2 = detect_landmarks(faceimg2)
        #二つの画像の特徴点の数が異なる場合エラー
        if len(points1) != len(points2):
            print('error: the numer of points are not same', file=sys.stderr)

        ##重み付けした2つの特徴点の中間点を決定
        points = np.empty((len(points1), 2))
        for j in range(len(points1)):
            points[j] = (points1[j]*(1-alpha) + points2[j]*alpha)

        output = np.zeros(img1.shape, dtype = img1.dtype)
        #ドロネーの三角分割を実行し、三角形を構成する特徴点のリストを取得
        tri = Delaunay(points1).simplices

        #対応する三角形同士をモーフィングしていく
        for k in range(len(tri)):
            p1, p2, p3 = tri[k][0], tri[k][1], tri[k][2]
            
            tri1 = np.float32([points1[p1], points1[p2], points1[p3]])
            tri2 = np.float32([points2[p1], points2[p2], points2[p3]])
            tri_dst = np.float32([points[p1], points[p2], points[p3]])

            # Morph one triangle at a time.
            morph_points(img1, img2, output, tri1, tri2, tri_dst, alpha)

        #画像を保存
        # output = np.uint8(output)
        # cv2.imwrite('output/%d_img.jpg' % i, output)
