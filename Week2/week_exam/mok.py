"""
Bài toán Max Over Kernel tìm giá trị lớn nhất,
trong một cửa sổ trượt trên danh sách cho trước
Ví dụ:
Input: num_list = [3, 4, 5, 1, -44 , 5 ,10, 12 ,33, 1] với k=3
Output: [5, 5, 5, 5, 10, 12, 33, 33]
k >= 1
"""
from collections import deque


def max_kernel(num_list, size):
    "Độ phức tạp về thời gian là O(n*k) với n phần tử * k kích thước cửa sổ "
    result = []
    for start in range(len(num_list)):
        if (len(num_list) - start) < size:
            return result
        else:
            slicing = num_list[start:start + size] 
            result.append(max(slicing))
    return result


def max_sliding_window(nums, k):
    """_summary_
    Tối ưu hơn do độ phức tạp về thời gian là O(n)
    Mô tả:
    - Phần tử bên trái của deq luôn là phần tử lớn nhất deq
    - Nếu có phần tử được thêm vào deq lớn hơn các phần tử hiện tại thì
    sẽ loại bỏ các phần tử hiện tại để thêm phần tử mới vào deq
    - Giải pháp như là một chú sâu, giản nở tối đa 3 phần tử và co lại tổi
    thiểu 0 phần tử, phần tử bên trái ngoài cùng luôn là lớn nhất. Thực hiện
    đến cuối list
    """
    if not nums or k == 0:
        return []
    deq = deque()
    result = []
    for i, num in enumerate(nums):
        # Loại bỏ các chỉ số nằm ngoài cửa sổ
        while deq and deq[0] <= i - k:
            deq.popleft()
        # Loại bỏ các phần tử nhỏ hơn num ở cuối deque
        while deq and nums[deq[-1]] <= num:
            deq.pop()
        deq.append(i)
        # Ghi nhận kết quả khi cửa sổ đủ k phần tử
        if i >= k - 1:
            result.append(nums[deq[0]])
    return result


num_list = [3, 4, 5, 1, -44, 5, 10, 12, 33, 1]
deq = deque(num_list)
size = 3
print(max_kernel(num_list, size))
print(max_sliding_window(num_list, size))
