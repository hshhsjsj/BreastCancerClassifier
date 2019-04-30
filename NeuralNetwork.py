import numpy as np
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape = (n_x, None), name = "X")
    Y = tf.placeholder(tf.int32, shape = (n_y, None), name = "Y")
    
    return X, Y

def initialize_parameters():
    W1 = tf.get_variable("W1", [100, 9], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [100, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [100, 100], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [100, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [100, 100], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [100, 1], initializer = tf.zeros_initializer())
    W4 = tf.get_variable("W4", [100, 100], initializer = tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", [100, 1], initializer = tf.zeros_initializer())
    W5 = tf.get_variable("W5", [2, 100], initializer = tf.contrib.layers.xavier_initializer())
    b5 = tf.get_variable("b5", [2, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5}
    
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
       
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.relu(Z3)
    Z4 = tf.add(tf.matmul(W4, A3), b4)
    A4 = tf.nn.relu(Z4)
    Z5 = tf.add(tf.matmul(W5, A4), b5)
    
    return Z5

def compute_cost(Z5, Y):
    logits = tf.transpose(Z5)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels))
    
    return cost
    
def plot_costs(costs):
    plt.plot(costs)
    plt.show()

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.000000001, num_epochs = 1000000):
    n_x, m = X_train.shape
    n_y = Y_train.shape[0]
    
    costs = []

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters()

    Z5 = forward_propagation(X, parameters)
    
    cost = compute_cost(Z5, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_global)
        sess.run(init_local)
            
        for i in range(num_epochs):
            _, epoch_cost = sess.run([optimizer, cost], feed_dict={X:X_train, Y:Y_train})
            costs.append(epoch_cost)
            if i%500==0:
                print("Epoch " + str(i) + ": Cost=" + str(epoch_cost))
                
        plot_costs(costs)
        
        parameters = sess.run(parameters)
        
        prediction = tf.argmax(Z5)
            
        correct_prediction = tf.math.equal(prediction, tf.cast(Y, "int64"))
        
        print("P", prediction.eval({X:X_test}))
        print("Y", Y.eval({Y:Y_test})[0])
        #print("C", correct_prediction.eval({X:X_test, Y:Y_test}))
        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train})*100, "%")
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test})*100, "%")
        print("Train f1:", sklearn.metrics.matthews_corrcoef(y_true = Y.eval({Y:Y_train})[0], y_pred = prediction.eval({X:X_train})))
        print("Test f1:", sklearn.metrics.matthews_corrcoef(y_true = Y.eval({Y:Y_test})[0], y_pred = prediction.eval({X:X_test})))
        print("Train AUC-ROC:", sklearn.metrics.roc_auc_score(y_true = Y.eval({Y:Y_train})[0], y_score = prediction.eval({X:X_train})))
        print("Test AUC-ROC:", sklearn.metrics.roc_auc_score(y_true = Y.eval({Y:Y_test})[0], y_score = prediction.eval({X:X_test})))
        
        return parameters
        
def split_train_test_set(train_test_split, dataset_size, x, labels):
    training_set_size = int(np.floor(train_test_split*dataset_size))

    x_train = (x[0:training_set_size] / 10.).transpose()
    x_test = (x[training_set_size:] / 10.).transpose()
    labels_train = (labels[0:training_set_size] // 4).transpose()
    labels_test = (labels[training_set_size:] // 4).transpose()
    
    return x_train, x_test, labels_train, labels_test
        
def get_csv_data(csv_directory, dataset_size, num_features, num_labels):
    data_x = np.empty((dataset_size, num_features), np.float32)
    data_labels = np.empty((dataset_size, num_labels), np.float32)
    
    file = open(csv_directory, "r")
    reader = csv.reader(file)
    count = 0
    for line in reader:
        data_x[count] = np.array([line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9]], np.int32)
        data_labels[count] = np.int32(line[10])
        count += 1
    
    return data_x, data_labels

def main():
    (x, labels) = get_csv_data("C:\\Users\\natha\\BreastCancerWisconsin\\raw_data.CSV", 699, 9, 1)
    (x_train, x_test, labels_train, labels_test) = split_train_test_set(0.9, 699, x, labels)
    final_parameters = model(x_train, labels_train, x_test, labels_test)
    print(final_parameters)

if __name__ == "__main__":
    main()