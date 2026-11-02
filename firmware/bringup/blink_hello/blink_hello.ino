// blink_hello.ino — milestone M1 step 2: first firmware on the board.
//
// Purpose: prove that the toolchain flashes the Arduino Nano 33 BLE Sense (Lite) and that the USB serial link works.
//   * LED_BUILTIN (orange, D13) blinks a distinctive pattern — three 80 ms flashes, then a pause — so the user can
//     tell this sketch from the factory Blink (steady 1 Hz).
//   * Once per second a heartbeat line "HB n=<count> ms=<uptime>" goes out over USB serial at 115200 baud.
//   * On start a banner with the sketch name and the CPU clock is printed; setup() waits at most 3 s for a host, so
//     the board also runs with nothing attached.
//
// Board: arduino:mbed_nano:nano33ble (core "Arduino Mbed OS Nano Boards" 4.6.0). No libraries needed.

#include <Arduino.h>

static uint32_t g_heartbeat = 0;

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  Serial.begin(115200);
  const uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 3000) {
    // wait for the host to open the port (DTR), but never block forever
  }
  Serial.println("HELLO kws-ua-nano33ble bringup blink_hello v1");
  Serial.print("BOARD nano33ble SystemCoreClock=");
  Serial.println(SystemCoreClock);
}

void loop() {
  // three short flashes: 80 ms on, 170 ms off  (3 x 250 ms = 750 ms), then rest until the second is over
  for (int i = 0; i < 3; ++i) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(80);
    digitalWrite(LED_BUILTIN, LOW);
    delay(170);
  }
  delay(250);

  ++g_heartbeat;
  Serial.print("HB n=");
  Serial.print(g_heartbeat);
  Serial.print(" ms=");
  Serial.println(millis());
}
