// (c) Michael Schoeffler 2017, http://www.mschoeffler.de

#include "Wire.h" // This library allows you to communicate with I2C devices.
#include "SD.h"

int CS = 4;
File file;
bool ende = false;

const int MPU_ADDR = 0x68; // I2C address of the MPU-6050. If AD0 pin is set to HIGH, the I2C address will be 0x69.

int16_t accelerometer_x, accelerometer_y, accelerometer_z; // variables for accelerometer raw data
int16_t gyro_x, gyro_y, gyro_z; // variables for gyro raw data
int16_t temperature; // variables for temperature data

unsigned long timestamp = millis();
unsigned long starttime;

char tmp_str[7]; // temporary variable used in convert function

char* convert_int16_to_str(int16_t i) { // converts int16 to string. Moreover, resulting strings will have the same length in the debug monitor.
  sprintf(tmp_str, "%6d", i);
  return tmp_str;
}

void setup() {
  Serial.begin(9600);
  Wire.begin();
  Wire.beginTransmission(MPU_ADDR); // Begins a transmission to the I2C slave (GY-521 board)
  Wire.write(0x6B); // PWR_MGMT_1 register
  Wire.write(0); // set to zero (wakes up the MPU-6050)
  Wire.endTransmission(true);

  pinMode(CS, OUTPUT);
  pinMode(7, OUTPUT);
  pinMode(9, INPUT);
  pinMode(8, INPUT);

  if (!SD.begin(CS)) {
    Serial.println("SD nicht initialisiert");  
  }

  file = SD.open("data.csv", FILE_WRITE);
  file.println("time,accX,accY,accZ,gyrX,gyrY,gyrZ");
}
void loop() {
  if(!ende){
    Serial.print("Warten auf Knopfdruck");
    while(digitalRead(9) == LOW){
      Serial.print(".");
      delay(50);
      if(digitalRead(8) == HIGH){
        Serial.println();
        Serial.println("Ende");
        ende = true;
        file.close();
        break;
      }  
    }
  }
  if(!ende){
    Serial.println();
    Serial.println("Aufnahme start");
    delay(2000);
    digitalWrite(7, HIGH);
    delay(50);
    digitalWrite(7, LOW);
    starttime = millis();
    while((millis()-starttime)<=3000){
      datamining();
    }
    digitalWrite(7, HIGH);
    delay(50);
    digitalWrite(7, LOW);
    Serial.println("Aufnahme ende");
  }

  
  // print out data
  /*Serial.print(timestamp);
  Serial.print(","); Serial.print(accelerometer_x);
  Serial.print(","); Serial.print(accelerometer_y);
  Serial.print(","); Serial.print(accelerometer_z);
  Serial.print(","); Serial.print(gyro_x);
  Serial.print(","); Serial.print(gyro_y);
  Serial.print(","); Serial.print(gyro_z);
  Serial.println();
  */
}

void datamining() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); // starting with register 0x3B (ACCEL_XOUT_H) [MPU-6000 and MPU-6050 Register Map and Descriptions Revision 4.2, p.40]
  Wire.endTransmission(false); // the parameter indicates that the Arduino will send a restart. As a result, the connection is kept active.
  Wire.requestFrom(MPU_ADDR, 7*2, true); // request a total of 7*2=14 registers
  
  // "Wire.read()<<8 | Wire.read();" means two registers are read and stored in the same variable
  timestamp = millis();
  accelerometer_x = Wire.read()<<8 | Wire.read(); // reading registers: 0x3B (ACCEL_XOUT_H) and 0x3C (ACCEL_XOUT_L)
  accelerometer_y = Wire.read()<<8 | Wire.read(); // reading registers: 0x3D (ACCEL_YOUT_H) and 0x3E (ACCEL_YOUT_L)
  accelerometer_z = Wire.read()<<8 | Wire.read(); // reading registers: 0x3F (ACCEL_ZOUT_H) and 0x40 (ACCEL_ZOUT_L)
  temperature = Wire.read()<<8 | Wire.read(); // reading registers: 0x41 (TEMP_OUT_H) and 0x42 (TEMP_OUT_L)
  gyro_x = Wire.read()<<8 | Wire.read(); // reading registers: 0x43 (GYRO_XOUT_H) and 0x44 (GYRO_XOUT_L)
  gyro_y = Wire.read()<<8 | Wire.read(); // reading registers: 0x45 (GYRO_YOUT_H) and 0x46 (GYRO_YOUT_L)
  gyro_z = Wire.read()<<8 | Wire.read(); // reading registers: 0x47 (GYRO_ZOUT_H) and 0x48 (GYRO_ZOUT_L)

  file.print(timestamp-starttime);
  file.print(","); file.print(accelerometer_x);
  file.print(","); file.print(accelerometer_y);
  file.print(","); file.print(accelerometer_z);
  file.print(","); file.print(gyro_x);
  file.print(","); file.print(gyro_y);
  file.print(","); file.print(gyro_z);
  file.println();
}
