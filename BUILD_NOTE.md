# 依赖安装

## Boost

``` bash
sudo apt update
sudo apt install build-essential python3 libbz2-dev libz-dev libicu-dev
wget https://archives.boost.io/release/1.89.0/source/boost_1_89_0.tar.gz
tar -xvf boost_1_89_0.tar.gz
cd boost_1_89_0
./bootstrap.sh --prefix=/usr/local
./b2
```

注意，完成 Boost 构建后，需要配置环境变量： ${BOOST_ROOT} 为 Boost 根目录。
