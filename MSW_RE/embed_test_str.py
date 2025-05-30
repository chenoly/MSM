import numpy as np
from models.msg import MSG
from matplotlib import pyplot as plt


print(np.__version__)
ACcode = MSG(box_size=24)
marked_code, emb_bits = ACcode.generate_MSG("hsdas", 0.5, 0.02)
# dm_mark_code = ACcode.add_marks(marked_code)
# cv2.imwrite("images/test.png", marked_code)
# cv2.imwrite("images/dm_mark_code.png", dm_mark_code)
ext_data, ext_bits = ACcode.Decode(marked_code)
print("decode :", ext_data)
print("embed__:", emb_bits)
print("extract:", ext_bits)
plt.imshow(marked_code, cmap='gray')
plt.show()
