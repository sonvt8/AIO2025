# Tạo kệ
def khoi_tao_ke():
    rows = int(input("Nhập số hàng của kệ: "))
    cols = int(input("Nhập số cột của kệ: "))
    ke = [[None for _ in range(cols)] for _ in range(rows)]
    return ke, rows, cols

# Nhập sản phẩm vào kệ
def nhap_san_pham(ke, rows, cols):
    print("Nhập tên sản phẩm cho từng vị trí trên kệ (nhập 'done' để kết thúc, để trống để giữ giá trị None):")
    for i in range(rows):
        for j in range(cols):
            san_pham = input(f"Nhập sản phẩm tại hàng {i+1} cột {j+1}: ").strip()
            if san_pham.lower() == 'done':
                return ke
            if san_pham:
                ke[i][j] = san_pham
            # Nếu chuỗi rỗng, giữ nguyên giá trị None, không cần thông báo
    return ke

# Tìm kiếm sản phẩm
def tim_san_pham(ke, rows, cols):
    while True:
        ten_san_pham = input("\nNhập tên sản phẩm cần tìm (nhập 'exit' để thoát): ").strip()
        if ten_san_pham.lower() == 'exit':
            print("Kết thúc tìm kiếm.")
            break
        
        found = False
        for i, row in enumerate(ke):
            for j, san_pham in enumerate(row):
                if san_pham and san_pham.lower() == ten_san_pham.lower():
                    print(f"{ten_san_pham} nằm ở vị trí hàng {i+1} cột {j+1}")
                    found = True
        if not found:
            print(f"Không tìm thấy sản phẩm {ten_san_pham} trên kệ.")

# Chương trình chính
def main():
    print("Chương trình quản lý sản phẩm trên kệ")
    ke, rows, cols = khoi_tao_ke()
    ke = nhap_san_pham(ke, rows, cols)
    print(ke)
    tim_san_pham(ke, rows, cols)

if __name__ == "__main__":
    main()