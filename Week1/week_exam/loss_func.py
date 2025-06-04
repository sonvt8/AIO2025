import random
import math

def cal_activation_function():
    """
    Hàm tính toán các loss functions: MAE, MSE, RMSE
    Theo đúng format yêu cầu của đề bài
    """
    
    # Input số lượng samples với validation
    while True:
        try:
            num_samples = int(input(">> Input number of samples (integer number) which are generated: "))
            if num_samples <= 0:
                print("number of samples must be an integer number")
                continue
            break
        except ValueError:
            print("number of samples must be an integer number")
    
    # Input loss name với validation
    valid_loss_names = ['MAE', 'MSE', 'RMSE']
    while True:
        loss_name = input("Input loss name: ").strip().upper()
        if loss_name in valid_loss_names:
            break
        else:
            print(f"loss name must be one of: {', '.join(valid_loss_names)}")
    
    # Generate random values và tính toán loss cho từng sample
    total_loss = 0
    
    for i in range(num_samples):
        # Generate random prediction và target values (0-10)
        pred_value = round(random.uniform(0, 10), 2)
        target_value = round(random.uniform(0, 10), 2)
        
        # Tính loss cho sample này
        if loss_name == 'MAE':
            loss_value = abs(target_value - pred_value)
        elif loss_name == 'MSE':
            loss_value = (target_value - pred_value) ** 2
        elif loss_name == 'RMSE':
            loss_value = (target_value - pred_value) ** 2
        
        total_loss += loss_value
        
        # In thông tin sample theo format yêu cầu
        print(f"loss_name: {loss_name}, sample: {i}: pred: {pred_value} target: {target_value} loss: {loss_value:.2f}")

    # Tính toán final loss
    if loss_name == 'RMSE':
        # Đối với RMSE, lấy căn bậc hai của MSE
        # final_loss = (total_loss / num_samples) ** 0.5
        final_loss = math.sqrt(total_loss / num_samples)
    else:
        # Đối với MAE và MSE, lấy trung bình
        final_loss = total_loss / num_samples
    
    print(f"final {loss_name}: {final_loss}")

# Chương trình chính
if __name__ == "__main__":
    cal_activation_function()
