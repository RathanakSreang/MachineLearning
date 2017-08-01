class SimpleLinearRegression
  attr_accessor :multiplier, :error

  def initialize(multiplier = 0.0, error = 0.0)
    self.multiplier = multiplier
    self.error = error
  end

  def fit(features, labels)
    calculate_multiplier_and_error(features, labels)
    puts "Multiplier: #{self.multiplier}"
    puts "Error: #{self.error}"
  end

  def predict(feature)
    return self.multiplier * feature + self.error
  end

  private

  def calculate_multiplier_and_error(features, labels)
    mean_features = mean(features)
    mean_labels = mean(labels)

    dividend = 0.0
    divisor = 0.0
    (0...labels.size).each do |index|
      # dividend = sum((features[i] -mean_features) * (labels[i] -mean_labels))
      dividend += (features[index] - mean_features) * (labels[index] - mean_labels)
      # divisor = sum((features[i] -mean_features) ** (features[i] -mean_features))
      divisor += (features[index] - mean_features) ** 2
    end
    # multiplier = dividend / divisor
    self.multiplier = dividend / divisor
    # error = mean_labels - self.multiplier * mean_features
    self.error = mean_labels - self.multiplier * mean_features
  end

  def mean(arr_data)
    arr_data.reduce(:+) / arr_data.size.to_f
  end
end

# load sample data
features = [1,2,4,3,5]
labels = [1,3,3,2,5]
puts "SimpleLinearRegression testing"
puts "Features: #{features}"
puts "Labels: #{labels}"
puts "-" * 50
puts "Training"
# init model
model = SimpleLinearRegression.new()
# train the model
model.fit(features, labels)
puts "-" * 50
puts "Prediction"
puts "Predict the known data"
# predict label by the given feature
features.each_with_index do |n, index|
  puts "x= #{n}, pred y: #{model.predict(n)}, actual y: #{labels[index]}"
end
puts "Predict the unknown data"
pred_label = model.predict(10)
puts "the predict of given feature 10 is: #{pred_label}"
