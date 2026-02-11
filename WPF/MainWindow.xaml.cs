using System;
using System.Collections.ObjectModel;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace HerbLight_Advisor
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        // 필드, 프로퍼티 --------------------------------------------------------------
        private const string PPFD_SUFFIX = "μmol";
        private const string HOUR_SUFFIX = "h";
        private const string MINUTE_SUFFIX = "m";
        private OnnxProcess _onnxProcess;
        public ObservableCollection<string> HerbNames { get; set; } = new ObservableCollection<string>();
        public double InputPpfd { get; set; }
        public double InputLightTime { get; set; }
        public double OutputDli { get; set; }

        public MainWindow()
        {
            InitializeComponent();

            _onnxProcess = new OnnxProcess();

            HerbNames.Add("기타");
            foreach(string name in _onnxProcess.HerbNames)
                HerbNames.Add(name);

            HerbList.ItemsSource = HerbNames;
            HerbList.SelectedIndex = 0;

            PpfdText.Text = "0.0" + PPFD_SUFFIX;
            LightTimeText.Text = "0" + HOUR_SUFFIX + " 0" + MINUTE_SUFFIX;
            DLIText.Text = "0.0%";
        }

        // 메소드 ----------------------------------------------------------------

        // 텍스트박스에 숫자만 입력 가능하도록 제한
        private bool IsTextNumeric(string text)
        {
            // 실수
            if (text == "." && text.Contains("."))
            {
                return true;
            }
            // 정수
            Regex regex = new Regex("[^0-9]+");
            return !regex.IsMatch(text);
        }

        private string ConvertPpfd(double input)
        {
            double truncated = Math.Truncate(input * 10) / 10.0;
            return truncated.ToString("0.0") + PPFD_SUFFIX;
        }

        private string ConvertLightTime(double input)
        {
            int inputHour = (int)input;
            int inputMinute = (int)((input - inputHour) * 60);
            return inputHour.ToString() + HOUR_SUFFIX + " " + inputMinute.ToString() + MINUTE_SUFFIX;
        }

        // 입력 이벤트 핸들러 ------------------------------------------------------

        // 슬라이더 - 텍스트박스 값 연동
        private void PpfdSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            InputPpfd = e.NewValue;
            PpfdText.Text = ConvertPpfd(InputPpfd);
        }

        private void LightTimeSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            InputLightTime = e.NewValue;
            LightTimeText.Text = ConvertLightTime(InputLightTime);
        }

        // 숫자만 입력되도록 제한
        private void Text_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            e.Handled = !IsTextNumeric(e.Text);
        }

        // 포커스를 받을 때 단위 제거
        private void Text_GotFocus(object sender, RoutedEventArgs e)
        {
            TextBox textbox = sender as TextBox;
            if (textbox != null)
            {
                if (PpfdText.IsFocused)
                {
                    string onlyNum = Regex.Replace(textbox.Text, PPFD_SUFFIX, "");
                    PpfdText.Text = onlyNum;
                }
                else if (LightTimeText.IsFocused)
                {
                    string onlyNum = LightTimeText.Text.Replace(HOUR_SUFFIX, "").Replace(MINUTE_SUFFIX, "").Trim();
                    LightTimeText.Text = onlyNum;
                }
                
                textbox.Dispatcher.BeginInvoke(new Action(() => textbox.SelectAll()), System.Windows.Threading.DispatcherPriority.Input);
            }
        }

        // 포커스를 잃을 때 슬라이더에 값 적용하고 단위 추가
        private void PpfdText_LostFocus(object sender, RoutedEventArgs e)
        {
            string text = PpfdText.Text.Trim();

            if (double.TryParse(text, out double value))
            {
                value = double.Min(2000, value);
                PpfdSlider.Value = value;
            }
            else
            {
                PpfdText.Text = ConvertPpfd(InputPpfd);
                Notice1.Text = "";
                InitBtn.Visibility = Visibility.Collapsed;
                Notice2.Text = "입력 형식이 잘못되었습니다. (정수.소수)";
            }
        }

        private void LightTimeText_LostFocus(object sender, RoutedEventArgs e)
        {
            string text = LightTimeText.Text;

            try
            {
                int hour;
                int minute;

                if (text.Contains(" "))
                {
                    int index = text.IndexOf(" ");
                    hour = Convert.ToInt32(text.Substring(0, index));
                    minute = Convert.ToInt32(text.Substring(index + 1));
                    
                    if (minute >= 60)
                    {
                        throw new Exception();
                    }
                }
                else
                {
                    hour = Convert.ToInt32(text);
                    minute = 0;
                }

                double value = hour + ((double)minute / 60);
                value = double.Min(18, value);
                LightTimeSlider.Value = value;
            }
            catch
            {
                LightTimeText.Text = ConvertLightTime(InputLightTime);
                Notice1.Text = "";
                InitBtn.Visibility = Visibility.Collapsed;
                Notice2.Text = "입력 형식이 잘못되었습니다. (시간 분 or 시간)";
            }
        }

        // 엔터키 입력시 입력 종료, 포커스 이동
        private void TextBox_EnterKeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Enter)
            {
                if (PpfdText.IsFocused)
                    LightTimeText.Focus();
                else if (LightTimeText.IsFocused)
                    ApplyBtn.Focus();
                
                e.Handled = true;
            }
        }

        // 출력 이벤트 핸들러 ------------------------------------------------------
        private void ApplyBtn_Click(object sender, RoutedEventArgs e)
        {
            float resultDLI = 0;
            if (HerbList.SelectedIndex == 0)
            {
                resultDLI = _onnxProcess.PredictAverageDLI((float)InputPpfd, (float)InputLightTime);
            }
            else
            {
                resultDLI = _onnxProcess.PredictDLI((float)InputPpfd, (float)InputLightTime, HerbList.SelectedIndex - 1);
            }
            DLISlider.Value = resultDLI;
            DLIText.Text = resultDLI.ToString("0.00") + "%";

            if (resultDLI > 200)
            {
                DLIText.Background = System.Windows.Media.Brushes.Red;
                DLIText.Foreground = System.Windows.Media.Brushes.LightGray;

                Notice1.Text = "심각한 과광량";
                Notice2.Text = "잎 화상 위험이 높으니 즉시 차광하거나 위치를 조정하세요.";
            }
            else if (resultDLI > 110)
            {
                DLIText.Background = System.Windows.Media.Brushes.OrangeRed;

                Notice1.Text = "과광량";
                Notice2.Text = "강광에 잘 적응하고 있는지 수시로 확인하세요.";
            }
            else if (resultDLI > 90)
            {
                DLIText.Background = System.Windows.Media.Brushes.Green;

                Notice1.Text = "적정광량";
                Notice2.Text = "현재 위치가 이상적이며, 안정적인 생육이 기대됩니다.";
            }
            else if (resultDLI > 70)
            {
                DLIText.Background = System.Windows.Media.Brushes.Blue;

                Notice1.Text = "저광량";
                Notice2.Text = "웃자람이 생길 수 있으니 더 밝은 곳을 고려하세요";
            }
            else
            {
                DLIText.Background = System.Windows.Media.Brushes.Navy;
                DLIText.Foreground = System.Windows.Media.Brushes.LightGray;

                Notice1.Text = "심각한 저광량";
                Notice2.Text = "장기적인 생존이 불가능하니 즉시 밝은 환경으로 옮기세요.";
            }

            if (InitBtn.Visibility != Visibility.Visible)
                InitBtn.Visibility = Visibility.Visible;
        }

        private void InitBtn_Click(object sender, RoutedEventArgs e)
        {
            PpfdSlider.Value = 0;
            LightTimeSlider.Value = 0;
            DLISlider.Value = 0;
            HerbList.SelectedIndex = 0;

            DLIText.Background = System.Windows.Media.Brushes.Transparent;
            DLIText.Foreground = System.Windows.Media.Brushes.Black;

            Notice1.Text = "";
            Notice2.Text = "";

            InitBtn.Visibility = Visibility.Collapsed;
        }
    }
}