num = "1"
switch(num,
"1" = setwd("../data/logs/search_result/test_data/"),
"2" = setwd("D:\\openCV_WorkSpace\\Ohno_cmake_Desktop\\build\\result01LH"),
"3" = setwd("/home/user/Guraduation_Resarch/Guraduation_Resarch/build/gra/Tsukuba02_locate"),
"4" = setwd("/home/user/Guraduation_Resarch/Guraduation_Resarch/build/result01LH"),
stop("only use 1 2 3 4")
)
getwd()

data_size <- switch(num, #データの取得 if 00 -> 670  if 01 -> 488
"1" = 67, 
"2" = 488, 
"3" = 4546, 
"4" = 488,
stop("error") 
)

x.data = c() #  空のベクトルを用意する．
y.data = c()
z.data = c()
u.data = c()
png1_name <- paste0("fv_cos_test_graph_color_left_black.png")#ここを自分の結果のファイルに合うように変えてね
png2_name <- paste0("fv_cos_test_graph_color_right_black.png")#ここを自分の結果のファイルに合うように変えてね
png3_name <- paste0("fv_cos_test_graph_dst_left_black.png")#ここを自分の結果のファイルに合うように変えてね
png4_name <- paste0("fv_cos_test_graph_dst_right_black.png")#ここを自分の結果のファイルに合うように変えてね

for (i in 0:data_size) {
  #類似度
  if (i < 10) {
    txt_name <- paste0("fv_cos_test_00000", i, ".csv") #ここを自分の結果のファイルに合うように変えてね
  } else if (i < 100) {
    txt_name <- paste0("fv_cos_test_0000", i, ".csv") #ここを自分の結果のファイルに合うように変えてね
  } else if (i < 1000) {
    txt_name <- paste0("fv_cos_test_000", i, ".csv") #ここを自分の結果のファイルに合うように変えてね
  } else if (i < 10000) {
    txt_name <- paste0("fv_cos_test_00", i, ".csv") #ここを自分の結果のファイルに合うように変えてね
  }
  txt_data <- read.table(txt_name, fileEncoding = "UTF-8", sep = ',')

  x.data <- c(x.data, rev(txt_data[[1]])) # ベクトルをつなげていく．
  y.data <- c(y.data, txt_data[[2]])
  z.data <- c(z.data, rev(txt_data[[3]]))
  u.data <- c(u.data, txt_data[[4]])
}

for (i in 0:data_size){
  ex <- c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
  dx <- x.data[(1+12*i):(11 + 12*i)]
  dy <- y.data[(2+12*i):(12 + 12*i)]
  dz <- z.data[(1+12*i):(11 + 12*i)]
  du <- u.data[(2+12*i):(12 + 12*i)]

  LM2x <- lm(dx ~ ex + I(ex^2), data=data.frame(ex, dx))
  LM2y <- lm(dy ~ ex + I(ex^2), data=data.frame(ex, dy))
  LM2z <- lm(dz ~ ex + I(ex^2), data=data.frame(ex, dz))
  LM2u <- lm(du ~ ex + I(ex^2), data=data.frame(ex, du))
  plot(ex, dx)
  abline(LM2x)
  print(summary(LM2x))
}

dx <- c()
dy <- c()
dz <- c()
du <- c()
for (i in 0:data_size){
  dx <- c(dx, x.data[(1+12*i):(11 + 12*i)])
  dy <- c(dy, y.data[(2+12*i):(12 + 12*i)])
  dz <- c(dz, z.data[(1+12*i):(11 + 12*i)])
  du <- c(du, u.data[(2+12*i):(12 + 12*i)])


}
ex = rep(seq(0, 1.0, 0.1), data_size+1)
LM2x <- lm(dx ~ ex + I(ex^2), data=data.frame(ex, dx))
LM2y <- lm(dy ~ ex + I(ex^2), data=data.frame(ex, dy))
LM2z <- lm(dz ~ ex + I(ex^2), data=data.frame(ex, dz))
LM2u <- lm(du ~ ex + I(ex^2), data=data.frame(ex, du))
print(LM2x)
print(LM2y)
print(LM2z)
print(LM2u)
