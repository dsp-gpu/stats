#include "all_test.hpp"
#include <core/services/console_output.hpp>

int main() {
  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Start();
  statistics_all_test::run();
  con.WaitEmpty();
  return 0;
}
