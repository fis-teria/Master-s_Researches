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
  #if (i == 1) {
  split.screen(c(2,2)) # 1行2列  
  screen(1)
  png(png1_name, width = 640, height = 640)
  par(mar = c(5, 5, 5, 10))
for (i in 0:data_size){
  if(i == 0){
    plot(0:10, x.data[(1+12*i):(11 + 12*i)], type = "l", ylim = c(-0.1, 1.0), xlab = "消去領域の割合", ylab = "コサイン類似度" , col = 4, pch = c(4),cex = 4)
  }else{
    par(new = T)
    plot(0:10, x.data[(1+12*i):(11 + 12*i)], type = "l", ylim = c(-0.1, 1.0), col = 4, pch = c(4),cex = 4, ann = F)
  }
}
  #par(xpd = T)
  #legend(par()$usr[2] + 0.6, par()$usr[4], legend = c("Now Location"), pch = c(4), col = c(4))

  screen(2)
  png(png2_name, width = 640, height = 640)
  par(mar = c(5, 5, 5, 10))
for (i in 0:data_size){
  if(i == 0){
    plot(0:10, y.data[(2+12*i):(12 + 12*i)], type = "l", ylim = c(-0.1, 1.0) , xlab = "消去領域の割合", ylab = "コサイン類似度" , col = 4, pch = c(4),cex = 4)
  }else{
    par(new = T)
    plot(0:10, y.data[(2+12*i):(12 + 12*i)], type = "l", ylim = c(-0.1, 1.0), col = 4, pch = c(4),cex = 4, ann = F)
  }

  #par(xpd = T)
  #legend(par()$usr[2] + 0.6, par()$usr[4], legend = c("Now Location"), pch = c(4), col = c(4))
}
  screen(3)
  png(png3_name, width = 640, height = 640)
  par(mar = c(5, 5, 5, 10))
for (i in 0:data_size){
  if(i == 0){
    plot(0:10, z.data[(1+12*i):(11 + 12*i)], type = "l", ylim = c(-0.1, 1.0) , xlab = "消去領域の割合", ylab = "コサイン類似度" , col = 4, pch = c(4),cex = 4)
  }else{
    par(new = T)
    plot(0:10, z.data[(1+12*i):(11 + 12*i)], type = "l", ylim = c(-0.1, 1.0), col = 4, pch = c(4),cex = 4, ann = F)
  }
}
  #par(xpd = T)
  #legend(par()$usr[2] + 0.6, par()$usr[4], legend = c("Now Location"), pch = c(4), col = c(4))

  screen(4)
  png(png4_name, width = 640, height = 640)
  par(mar = c(5, 5, 5, 10))
for (i in 0:data_size){
  if(i == 0){
    plot(0:10, u.data[(2+12*i):(12 + 12*i)], type = "l", ylim = c(-0.1, 1.0) , xlab = "消去領域の割合", ylab = "コサイン類似度" , col = 4, pch = c(4),cex = 4)
  }else{
    par(new = T)
    plot(0:10, u.data[(2+12*i):(12 + 12*i)], type = "l", ylim = c(-0.1, 1.0), col = 4, pch = c(4),cex = 4, ann = F)
  }

  #par(xpd = T)
  #legend(par()$usr[2] + 0.6, par()$usr[4], legend = c("Now Location"), pch = c(4), col = c(4))
}
