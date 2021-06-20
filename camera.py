import cv2
import torch
import numpy as np
from neural_style.transformer_net import TransformerNet

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)


transformer = TransformerNet().cuda()
transformer.load_state_dict(torch.load('./output_model_dir/final_model.pth'))
transformer.eval()

while(True):
    ret, frame = cap.read()
    print('frame', frame.shape)
    input_frame = torch.tensor(frame).float().unsqueeze(0)
    input_frame = input_frame.permute(0, 3, 1, 2).cuda() 

    stylized_frame = transformer(input_frame)
    stylized_frame = stylized_frame.permute(0, 2, 3, 1).detach().cpu()
    stylized_frame = torch.clamp(stylized_frame, min=0.0, max=255.0).int().numpy()

    stylized_frame = stylized_frame[0].astype(np.uint8)

    cv2.imshow('frame', stylized_frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()