@echo off
"C:\\Users\\ASUS\\AppData\\Local\\Android\\Sdk\\cmake\\3.22.1\\bin\\cmake.exe" ^
  "-HD:\\TA Skripsi\\aplikasi\\HeightMeasurement\\sdk\\libcxx_helper" ^
  "-DCMAKE_SYSTEM_NAME=Android" ^
  "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON" ^
  "-DCMAKE_SYSTEM_VERSION=26" ^
  "-DANDROID_PLATFORM=android-26" ^
  "-DANDROID_ABI=x86_64" ^
  "-DCMAKE_ANDROID_ARCH_ABI=x86_64" ^
  "-DANDROID_NDK=C:\\Users\\ASUS\\AppData\\Local\\Android\\Sdk\\ndk\\27.0.12077973" ^
  "-DCMAKE_ANDROID_NDK=C:\\Users\\ASUS\\AppData\\Local\\Android\\Sdk\\ndk\\27.0.12077973" ^
  "-DCMAKE_TOOLCHAIN_FILE=C:\\Users\\ASUS\\AppData\\Local\\Android\\Sdk\\ndk\\27.0.12077973\\build\\cmake\\android.toolchain.cmake" ^
  "-DCMAKE_MAKE_PROGRAM=C:\\Users\\ASUS\\AppData\\Local\\Android\\Sdk\\cmake\\3.22.1\\bin\\ninja.exe" ^
  "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=D:\\TA Skripsi\\aplikasi\\HeightMeasurement\\sdk\\build\\intermediates\\cxx\\Debug\\553t6e5g\\obj\\x86_64" ^
  "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=D:\\TA Skripsi\\aplikasi\\HeightMeasurement\\sdk\\build\\intermediates\\cxx\\Debug\\553t6e5g\\obj\\x86_64" ^
  "-DCMAKE_BUILD_TYPE=Debug" ^
  "-BD:\\TA Skripsi\\aplikasi\\HeightMeasurement\\sdk\\.cxx\\Debug\\553t6e5g\\x86_64" ^
  -GNinja ^
  "-DANDROID_STL=c++_shared"
