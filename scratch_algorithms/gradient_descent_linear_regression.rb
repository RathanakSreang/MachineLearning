class GradientDescentLinearRegression
  attr_accessor :multiplier, :error, :alpha, :epochs

  def initialize(alpha = 0.01, epochs = 10, multiplier = 0.0, error = 0.0)
    self.alpha = alpha
    self.epochs = epochs
    self.multiplier = multiplier
    self.error = error
  end

  def fit(features, labels)
    update_weights(features, labels)
    puts "alpha: #{self.alpha}"
    puts "epochs: #{self.epochs}"
    puts "multiplier: #{self.multiplier}"
    puts "error: #{self.error}"
  end

  def predict(feature)
    return self.multiplier * feature + self.error
  end

  private

  def update_weights(features, labels)
    self.epochs.times do
      (0...labels.size).each do |index|
        # calculate the prediction
        predicted = self.multiplier * features[index] + self.error

        # calculate the error
        err = predicted - labels[index]

        # update multiplier and error
        self.error =  self.error - alpha * err
        self.multiplier = self.multiplier - alpha * err * features[index]
      end
    end
  end
end

# load sample data
features = [1,2,4,3,5]
labels = [1,3,3,2,5]
puts "GradientDescentLinearRegression testing"
puts "Features: #{features}"
puts "Labels: #{labels}"
puts "-" * 50
puts "Training"
# init model
model = GradientDescentLinearRegression.new()
# train the model
model.fit(features, labels)
# predict label by the given feature
puts "Predict the known data"
features.each_with_index do |n, index|
  puts "x= #{n}, pred y: #{model.predict(n)}, actual y: #{labels[index]}"
end
puts "Predict the unknown data"
pred_label = model.predict(10)
puts "-" * 50
puts "Prediction"
puts "the predict of given feature 10 is: #{pred_label}"
