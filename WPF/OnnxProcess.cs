using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json.Linq;

namespace HerbLight_Advisor
{
    public class OnnxProcess : IDisposable
    {
        private InferenceSession _session;
        private JObject _metadata;

        public List<string> HerbNames { get; private set; }

        // 입출력 범위 - 역정규화에서 활용
        private float _ppfdMin;
        private float _ppfdMax;
        private float _timeMin;
        private float _timeMax;
        private float[] _outputMin;
        private float[] _outputMax;

        public OnnxProcess()
        {
            string modelPath = @"..\..\..\..\AI\HerbLight_Advisor_model.onnx";
            string jsonPath = @"..\..\..\..\AI\scaler_meta_info.json";

            try
            {
                // JSON 메타데이터 로드
                string jsonContent = File.ReadAllText(jsonPath);
                _metadata = JObject.Parse(jsonContent);

                // 식물 이름 추출
                HerbNames = new List<string>();
                var herbsArray = _metadata["plant_names"] as JArray;
                foreach (var herb in herbsArray)
                {
                    HerbNames.Add(herb.ToString());
                }

                // 입력 범위 추출
                var inputRanges = _metadata["input"] as JObject;
                _ppfdMin = inputRanges["light_min"].Value<float>();
                _ppfdMax = inputRanges["light_max"].Value<float>();
                _timeMin = inputRanges["time_min"].Value<float>();
                _timeMax = inputRanges["time_max"].Value<float>();

                // 출력 범위 추출
                var outputRanges = _metadata["output"] as JObject;
                _outputMin = new float[HerbNames.Count];
                _outputMax = new float[HerbNames.Count];

                for (int i = 0; i < HerbNames.Count; i++)
                {
                    string herbName = HerbNames[i];
                    _outputMin[i] = outputRanges[herbName]["min"].Value<float>();
                    _outputMax[i] = outputRanges[herbName]["max"].Value<float>();
                }

                // ONNX 모델 로드
                _session = new InferenceSession(modelPath);
            }
            catch (Exception ex)
            {
                throw new Exception($"ONNX 초기화 실패: {ex.Message}", ex);
            }
        }

        // Min-Max 정규화 (0~1)
        private float Normalize(float value, float min, float max)
        {
            if (max == min) return 0f;
            return (value - min) / (max - min);
        }

        // 역정규화 (0~1 -> 실제 값)
        private float Denormalize(float normalized, float min, float max)
        {
            return normalized * (max - min) + min;
        }

        // 특정 식물에 대한 DLI 예측
        public float PredictDLI(float ppfd, float time, int plantIndex)
        {
            if (plantIndex < 0 || plantIndex >= HerbNames.Count)
                throw new ArgumentOutOfRangeException(nameof(plantIndex));

            // 입력 정규화
            float normalizedPpfd = Normalize(ppfd, _ppfdMin, _ppfdMax);
            float normalizedTime = Normalize(time, _timeMin, _timeMax);

            // 입력 텐서 생성 [1, 2]
            var inputTensor = new DenseTensor<float>(new[] { 1, 2 });
            inputTensor[0, 0] = normalizedPpfd;
            inputTensor[0, 1] = normalizedTime;

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            // 추론 실행
            using (var results = _session.Run(inputs))
            {
                var output = results.First().AsEnumerable<float>().ToArray();

                // 선택된 식물의 출력값 추출 및 역정규화
                float normalizedDli = output[plantIndex];
                float actualDli = Denormalize(normalizedDli, _outputMin[plantIndex], _outputMax[plantIndex]);

                return actualDli;
            }
        }

        // 모든 식물의 평균 DLI 예측
        public float PredictAverageDLI(float ppfd, float time)
        {
            float sum = 0f;

            for (int i = 0; i < HerbNames.Count; i++)
            {
                sum += PredictDLI(ppfd, time, i);
            }

            return sum / HerbNames.Count;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
