library(tidyverse)
library(keras)
library(kerasR)
library(tidyverse)
library(knitr)
#To prevent the local unbound variabl error
library(tensorflow)
install_tensorflow(version="1.12")
mnist<-dataset_mnist()
cifar10 <- dataset_cifar10()

install_tensorflow(version = "1.12") 
install_tensorflow(version = "1.12") install_tensorflow(version = "1.12") ##loading training and test data into separate variables
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test  <- mnist$test$x
y_test  <- mnist$test$y

#flattening training and test data to remove spatial relationships
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
x_test <-  array_reshape(x_test, c(nrow(x_test), 784))
dim(x_train)
#rescalaing the above from grayscale to floating point between 0 and 1

x_train <- x_train /255
x_test  <- x_test/255

#one-hot encoding the vectors in y into binary classes
y_train <- to_categorical(y_train, 10)
y_test  <- to_categorical(y_test, 10)

#defining model using sequential model here

model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, input_shape = c(784)) %>%
  layer_dense(units = 10, activation = 'softmax')
summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history <- model %>% fit(
  x_train, y_train,
  batch_size =128,
  epochs = 12,
  verbose = 1,
  validation_split = 0.2
)
plot(history)

history %>% as_tibble() %>% kable
score <- model %>% evaluate(
  x_test, y_test,
  verbose = 0
)
score
score %>% as_tibble() %>% kable 
  
##New model for Question 8

modelRelu <- keras_model_sequential()
modelRelu %>%
  layer_dense(units = 256, activation ='relu', input_shape = c(784)) %>%
  layer_dense(units = 10, activation = 'softmax')
summary(model)

modelRelu %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

historyRelu <- modelRelu %>% fit(
  x_train, y_train,
  batch_size =128,
  epochs = 12,
  verbose = 1,
  validation_split = 0.2
)
plot(historyRelu, title(main = "relu"))

scoreRelu <- modelRelu %>% evaluate(
  x_test, y_test,
  verbose = 0
)
scoreRelu
scoreRelu %>% as_tibble() %>% kable 



#Reshape to data to fit model Question 11

x_train_new = mnist$train$x
x_test_new = mnist$test$x
y_train_new = mnist$train$y
y_test_new = mnist$test$y

x_train_new <- array_reshape(x_train_new, c(nrow(x_train_new), 28, 28, 1))
dim(x_train_new)
x_test_new <- array_reshape(x_test_new, c(nrow(x_test_new), 28, 28, 1))
dim(x_test_new)
x_train_new = x_train_new/255
x_test_new = x_test_new/255


y_train_new <- to_categorical(y_train_new, 10)
y_test_new  <- to_categorical(y_test_new, 10)

modelDeep <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),activation = 'relu', input_shape = c(28,28,1)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3),activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')
summary(modelDeep)

modelDeep %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

historyDeep<- modelDeep %>% fit(
  x_train_new, y_train_new,
  batch_size =128,
  epochs = 6,
  verbose = 1,
  validation_split = 0.2
)
plot(historyDeep)
scoreDeep <- modelDeep %>% evaluate(
  x_test_new, y_test_new,
  verbose = 0
)
scoreDeep
scoreDeep %>% as_tibble() %>% kable 

  
modelDeepDrop <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),activation = 'relu', input_shape = c(28,28,1)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3),activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = 'softmax')
summary(modelDeepDrop)

modelDeepDrop %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

historyDeepDrop<- modelDeepDrop %>% fit(
  x_train_new, y_train_new,
  batch_size =128,
  epochs = 6,
  verbose = 1,
  validation_split = 0.2
)
plot(historyDeepDrop)
scoreDeepDrop <- modelDeepDrop %>% evaluate(
  x_test_new, y_test_new,
  verbose = 0
)

scoreDeepDrop
scoreDeepDrop %>% as_tibble() %>% kable 



#Question 18 to 22
x_train_cifar <- cifar10$train$x
x_test_cifar <- cifar10$test$x
y_train_cifar <- cifar10$train$y
y_test_cifar <- cifar10$test$y


x_train_cifar <- x_train_cifar/255
x_test_cifar <- x_test_cifar/255
y_train_cifar <- to_categorical(y_train_cifar, 10)
y_test_cifar <- to_categorical(y_test_cifar, 10)


modelDeepDropCifar <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),activation = 'relu', input_shape = c(32, 32, 3), padding = 'same') %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),activation = 'relu', padding = 'same') %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3),activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = 'softmax')
summary(modelDeepDropCifar)

modelDeepDropCifar %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = c('accuracy')
)

historyDeepDropCifar<- modelDeepDropCifar %>% fit(
  x_train_cifar, y_train_cifar,
  batch_size =32,
  epochs = 20,
  verbose = 1,
  validation_data = list(x_test_cifar, y_test_cifar),
  validation_split = 0.2,
  shuffle = TRUE
)
plot(historyDeepDropCifar)
scoreDeepDropCifar <- modelDeepDropCifar %>% evaluate(
  x_test_cifar, y_test_cifar,
  verbose = 0
)

scoreDeepDropCifar
scoreDeepDropCifar %>% as_tibble() %>% kable 
