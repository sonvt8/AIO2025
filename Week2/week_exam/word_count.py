import os
import gdown
import re

def download_file(f, url):
    # Kiểm tra file đã tồn tại chưa
    if os.path.exists(f):
        pass
    else:
        print("File chưa tồn tại, tiến hành tải về.")
        gdown.download(url, f, quiet=False)

    return open(f, 'r').read()

def count_word(content:str):
    re.sub(r'[^a-zA-Z0-9\s-]', '', content)
    words = content.lower().split()
    result = {}
    
    for word in words:
        result[word] = result.get(word, 0) + 1
        
    return result
        
    
def main():
    file_path = "downloaded_file.txt"
    url = "https://drive.google.com/uc?id=1IBScGdW2xlNsc9v5zSAya548kNgiOrko"
    content = download_file(file_path, url)
    wc = count_word(content)
    
    assert wc["who"] == 3
    print(wc["man"])
    

if __name__ == "__main__":
    main()