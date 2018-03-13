#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#define BNO055_SAMPLERATE_DELAY_MS (100)

Adafruit_BNO055 bno = Adafruit_BNO055(55);
    
void setup() {
  // put your setup code here, to run once:
      Serial.begin(115200);
      Serial.println("Orientation Sensor Test"); Serial.println("");
      
      /* Initialise the sensor */
      if(!bno.begin())
      {
        /* There was a problem detecting the BNO055 ... check your connections */
        Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
        while(1);
      }
      
      delay(1000);
      bno.setExtCrystalUse(true);
}

void loop() {
      imu::Vector<3> linaccel = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL); // !!!!

      /* Display the floating point data */
      Serial.print("X: ");
      Serial.print(linaccel.x(),4);
      Serial.print("\tY: ");
      Serial.print(linaccel.y(),4);
      Serial.print("\tZ: ");
      Serial.print(linaccel.z(),4);
      Serial.println("");

  delay(BNO055_SAMPLERATE_DELAY_MS);



}
