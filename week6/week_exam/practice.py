from matplotlib.pyplot import cla
from naiveBayes import NaiveBayesClassifier
from gaussionPDF import GaussianPDF

def main():
  classifier = NaiveBayesClassifier()
  
  # Dữ liệu huấn luyện
  train_data = classifier.create_training_data("data1.xlsx")

  # Dữ liệu test
  X = ['Sunny', 'Cool', 'High', 'Strong']
  
  # Huấn luyện mô hình
  classifier.train(train_data)
  
  # Kết quả các tham số
  prior_prob = classifier._prior_probs
  class_names = classifier._class_names
  feature_values = classifier._feature_names
  class_probabilities = classifier._class_probabilities
  prediction, prob_dict = classifier.predict_result(X)
  
  # Câu 1
  print("**"*20)
  print("Đáp án câu 1:")
  print(f'P("Play Tennis" = {class_names[0]}):', prior_prob[0])
  print(f'P("Play Tennis" = {class_names[1]}):', prior_prob[1])
  
  # Câu 2
  print("**"*20)
  print("Đáp án câu 2:")
  print(f'P("Play Tennis" = "Yes"|X là {class_probabilities[1]})')
  
  # Câu 3
  print("**"*20)
  print("Đáp án câu 3:")
  print(f'P("Play Tennis" = "No"|X là {class_probabilities[0]})')
  
  # Câu 4
  print("**"*20)
  print("Đáp án câu 4:")
  print(f'P("Play Tennis = {prediction}")')
  
  # Câu 5
  print("**"*20)
  print("Đáp án câu 5:")
  print("Xác suất tiên nghiệm là:")
  print(f'P("Play Tennis" = {class_names[0]}):', prior_prob[0])
  print(f'P("Play Tennis" = {class_names[1]}):', prior_prob[1])
  
  # Câu 6
  print("**"*20)
  print("Đáp án câu 6:")
  for id, value in enumerate(feature_values):
    print(f"x{id + 1} = {value}")
    
  # Câu 7
  print("**"*20)
  print("Đáp án câu 7:")
  outlook = feature_values[0]
  i1 = classifier.get_feature_index("Overcast", outlook)
  i2 = classifier.get_feature_index("Rain", outlook)
  i3 = classifier.get_feature_index("Sunny", outlook)
  print(i1, i2, i3)
  
  # Câu 8
  print("**"*20)
  print("Đáp án câu 8:")
  print("Ad should not go!") if prediction == "No" else  print("Ad should go!")
  
  print("==="*20)
  print("Nạp dữ liệu mới")
  classifier_new = NaiveBayesClassifier()
  # Dữ liệu huấn luyện mới
  train_data_2 = classifier_new.create_training_data("data2.xlsx")

  # Huấn luyện mô hình
  classifier_new.train(train_data_2)
  
  # Dữ liệu test mới
  X2 = ['Weekday', 'Winter', 'High', 'Heavy']
  
  # Kết quả các tham số mới
  prior_prob = classifier_new._prior_probs
  class_names = classifier_new._class_names
  feature_values = classifier_new._feature_names
  prediction, _ = classifier_new.predict_result(X2)

  # Câu 9
  print("**"*20)
  print("Đáp án câu 9:")
  for idx, value in enumerate(class_names):
    print(f'P("Class" = {value}) = {prior_prob[idx]}')

  # Câu 10
  print("**"*20)
  print("Đáp án câu 10:")
  prob_x = round(classifier_new.predict_result_x("On Time"),4)
  print(f'P("Class" = "On Time" | X) ∝ {prob_x}')
  
  # Câu 11
  print("**"*20)
  print("Đáp án câu 11:")
  prob_x = round(classifier_new.predict_result_x("Late"),4)
  print(f'P("Class" = "Late" | X) ∝ {prob_x}')
  
  # Câu 12
  print("**"*20)
  print("Đáp án câu 12:")
  prob_x = round(classifier_new.predict_result_x("Very Late"),4)
  print(f'P("Class" = "Very Late" | X) ∝ {prob_x}')
  
  # Câu 13
  print("**"*20)
  print("Đáp án câu 13:")
  prob_x = round(classifier_new.predict_result_x("Cancelled"),4)
  print(f'P("Class" = "Cancelled" | X) ∝ {prob_x}')
  
  # Câu 14
  print("**"*20)
  print("Đáp án câu 14:")
  print(f'Dự đoán "Class"của sự kiện X là: {prediction}')
  
  # Câu 15 & 16
  print("**"*20)
  print("Đáp án câu 15 và 16:")
  model = GaussianPDF()
  model.create_training_data("data3.xlsx", feature_col="Length",
                            class_col="Class")
  model.print_class_stats()
  
  # Câu 17
  print("**"*20)
  print("Đáp án câu 17:")
  p = model.gaussian_pdf(3.4, class_label=0)
  print(f"P(x=3.4 | class=0) = {p:.6f}")
  p = model.gaussian_pdf(3.4, class_label=1)
  print(f"P(x=3.4 | class=1) = {p:.6f}")
  
if __name__ == "__main__":
    main()