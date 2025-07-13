from naiveBayes import NaiveBayesClassifier

def main():
  classifier = NaiveBayesClassifier()
  
  # Dữ liệu huấn luyện
  train_data = classifier.create_training_data("data1.xlsx")
  # print(train_data)
  
  # Dữ liệu test
  X = ['Sunny', 'Cool', 'High', 'Strong']
  
  # Huấn luyện mô hình
  classifier.train(train_data)
  
  # Kết quả các tham số
  prior_prob = classifier._prior_probs
  class_names = classifier._class_names
  feature_values = classifier._feature_names
  prediction, prob_dict, class_probabilities = classifier.predict_tennis(X)
  
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
  
if __name__ == "__main__":
    main()