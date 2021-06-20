import cv2
import os
from datetime import datetime



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)

train_dir = 'train_dataset'
os.makedirs(train_dir, exist_ok=True)



while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) 
    if key == ord('q'):
        break
    elif key == ord('p'):
        now = datetime.now()
        filename = now.strftime("%H_%M_%S") + '.jpg'
        filename = os.path.join(train_dir, filename)
        print('Save Image To Dir: ', filename)
        cv2.imwrite(filename, frame)
cap.release()
cv2.destroyAllWindows()