"use client"

import React, { useState } from 'react';
import { ResponsiveLine, Serie } from '@nivo/line';
import Papa from 'papaparse';
import { useSearchParams } from 'next/navigation';
import Select from 'react-select';


const Playground = () => {
  window.addEventListener("beforeunload", function(e){
    e.preventDefault();
 }, false);
  const isDark = localStorage.getItem("mode") == "Dark";
  const isExistResults = {
    exist: 1,
    notExist: 2,
    training: 3
  }
  const [text, setText] = useState("");
  const [numOfExamples, setNumOfExamples] = useState("");
  const [result, setResult] = useState<any>(null);
  const [reliabilities, setReliabilities] = useState<any>();

  const [currentArch, setCurrentArch] = useState<string>("GRU");
  const [isLoading, setIsLoading] = useState(false);
  
  const [existResult, setExistResult] = useState(0);
  const [trainCustom, setTrainCustom] = useState(false);

  const emotions = [
    {value: "happy", label: "Happinnes"},
    {value: "sadness", label: "Sadness"},
    {value: "anger", label: "Anger"},
    {value: "admiration", label: "Admiration"},
    {value: "suprise", label: "Suprise"},
    {value: "worry", label: "Worry"},
  ]

  const [emotionsChosen, setEmotionsChosen] = useState<any>([emotions[0].value, emotions[1].value, emotions[2].value]);

  const emotionsSize = {
    "happy": 21000,
    "sadness": 18000,
    "anger": 11000,
    "admiration": 17000,
    "suprise": 14000,
    "worry": 10000
  }
  
  const countMaxExamples = (feelings: Array<string>) => {
    let sum = 0;
    for (let i in feelings) {
      sum += emotionsSize[feelings[i]];
    }
    return sum;
  } 

  const architectureChosen = (e:any) => {
    setCurrentArch(e.target.innerText);
    checkIfModelExist(e.target.innerText, emotionsChosen);
  }

  const checkIfModelExist = async (currentArch:string, feelings:Array<string>) => {
    const res = await fetch("http://localhost:8080/is_exist?arch=" + currentArch + "&feelings=" + JSON.stringify(feelings));
    const responseJson = await res.json();
    setExistResult(responseJson.res.is_exist)
   } 

  // const parseData = () =>{
  //   let array : Serie[] = [];

  //   Papa.parse("/assets/GRU-accuracy.csv", {
  //       download: true,
  //       complete: function(results) {
  //         const data : any = results.data;
          
  //         let final = []
  //         for(let i = 1; i < data.length - 1; i++) {
  //           final.push({x: i, y: data[i][1]});
  //         }
  //         const nivoData = {
  //           id: "GRU",
  //           color: "hsl(54, 70%, 50%)",
  //           data: final
  //         }
  //         array.push(nivoData);
  //       }
  //   });
  //   Papa.parse("/assets/LSTM-accuracy.csv", {
  //     download: true,
  //     complete: function(results) {
  //       const data : any = results.data;
        
  //       let final = []
  //       for(let i = 1; i < data.length - 1; i++) {
  //         final.push({x: i, y: data[i][1]});
  //       }
  //       const nivoData = {
  //         id: "LSTM",
  //         color: "hsl(141, 70%, 50%)",
  //         data: final
  //       }
  //       array.push(nivoData);
  //     }
  //   });
  //   console.log(array);
  //   return array;
  // }

  const [data, setData] = useState<Serie[]>([]);

  const click = async () => {
    setIsLoading(true);
    setResult(0);
    setReliabilities(null);
    
    let array : Serie[] = [];
    try {
      Papa.parse("/assets/"+currentArch+"-"+JSON.stringify(emotionsChosen)+"accuracy.csv", {
        download: true,
        complete: function(results) {
          const data : any = results.data;
          
          let final = []
          for(let i = 1; i < data.length - 1; i++) {
            final.push({x: i, y: data[i][1]});
          }
          const nivoData = {
            id: currentArch,
            color: "hsl(54, 70%, 50%)",
            data: final
          }
          array.push(nivoData);
          setData(array);
        }
      });
    } catch (error) {
      
    }
    
    const anotherRes = await fetch("http://localhost:8080/sentiment?arch=" + currentArch + "&feelings=" + JSON.stringify(emotionsChosen)+ "&sentence=" + text);
    const json = await anotherRes.json();
    if(json?.res) {
      let reliabilities:any = [];

      Object.keys(json.res?.reliability).forEach((key) => {
        console.log(key, json.res?.reliability[key])
        reliabilities.push(<p>{key}: {json.res?.reliability[key]}</p>)
      });
      setReliabilities(reliabilities);

      setResult(json.res);
    }
  }

  return (
    <>
      <div className="w-full h-full fixed z-[999] flex items-center justify-center pointer-events-none">
        {
          trainCustom &&
          <div className="w-1/3 h-1/4 bg-white pointer-events-auto flex flex-col gap-2 items-center justify-center" style={isDark ? {color: "white", backgroundColor: "black", border: "1px solid white"} : {border: "1px solid black"}}>
            <h1>Train Custom Model</h1>
            <p>Architecture: {currentArch}, Feelings: {JSON.stringify(emotionsChosen)}</p>
            <input className='rounded-md p-2 pointer-events-auto' style={isDark ? {color: "white", backgroundColor: "black", border: "1px solid white"} : {border: "1px solid black"}} placeholder={'Number of examples: ' + countMaxExamples(emotionsChosen)} onChange={(e) => setNumOfExamples(e.target.value)}/>
            <h3 onClick={async () => {
              setTrainCustom(false);
              const anotherRes = await fetch("http://localhost:8080/train-custom?arch=" + currentArch + "&feelings=" + JSON.stringify(emotionsChosen)+ "&num_of_examples=" + numOfExamples);
              console.log(anotherRes.json());
            }} className="p-2 cursor-pointer pointer-events-auto" style={isDark ? {color: "white", backgroundColor: "black", border: "1px solid white"} : {border: "1px solid black"}}>Train</h3>
          </div>
        }
      </div>
      <div className='h-full w-full flex flex-row-reverse items-center justify-center gap-2 font-[Poppins] z-50' style={isDark ? { backgroundColor: 'black' } : { backgroundColor: 'white' }}>
        {/* <p>{text}</p> */}
        <div className='flex flex-col justify-center items-center gap-2 h-1/2 w-fit px-52 border-l-solid border-l-white border-l-[1px]' style={isDark ? {borderLeft: "1px solid white"} : {borderLeft: "1px solid black"}}>
          <h1 className='text-2xl font-bold' style={isDark ? {color: "white"} : {}}>Playground</h1>
          <input className='rounded-md p-2' style={isDark ? {color: "white", backgroundColor: "black", border: "1px solid white"} : {border: "1px solid black"}} placeholder='Text' onChange={(e) => setText(e.target.value)}/>

          <div className='flex gap-2'>
            <button onClick={architectureChosen} className="border-solid border-[1px] border-black rounded-md p-2" style={currentArch == "GRU" ? {backgroundColor: "rgba(211,211,211,1)"} : {backgroundColor: "white"}}>GRU</button>
            <button onClick={architectureChosen} className="border-solid border-[1px] border-black rounded-md p-2" style={currentArch == "LSTM" ? {backgroundColor: "rgba(211,211,211,1)"} : {backgroundColor: "white"}}>LSTM</button>
          </div>

          <Select 
            options={emotions}
            isMulti
            isSearchable
            isClearable
            closeMenuOnSelect={false}
            onChange={(newValue, actionMeta) => {
              let _emotionsChosen : any = [];
              newValue.forEach((value) => {
                _emotionsChosen.push(value.value);
              });
              setEmotionsChosen(_emotionsChosen);
              console.log(_emotionsChosen);
              checkIfModelExist(currentArch, _emotionsChosen);
            }}
            name="emotions"
            isLoading={isLoading}
            defaultValue={[emotions[0], emotions[1], emotions[2]]}
          />
          
          {existResult == isExistResults.exist && <p className='text-white text-center'>This model exists! ✅ You're welcome to test it :)</p>}
          {existResult == isExistResults.notExist && <p className='text-white text-center'>No one has made such model ❌ - <a className='underline cursor-pointer' onClick={() => { setTrainCustom(true); }}>Create a new model now!</a></p>}
          {existResult == isExistResults.training && <p className='text-white text-center'>What a cooincidence! 🛠 This model is currently in a training session and will soon be usable :(</p>}
            
          <button className='rounded-xl p-2 font-medium' style={isDark ? {color: "white", backgroundColor: "black", border: "3px solid white"} : {border: "3px solid rgb(219 219 219)", backgroundColor: "#f1f3f4"}} onClick={click}>Get Result</button>

          <div className='p-2' style={isDark ? {color: "white", backgroundColor: "black", border: "1px solid white"} : {border: "2px solid rgb(219 219 219)"}}>
            {result != null && result != 0 && <>
              Results:
              <p>Sentence: {result?.sentence}</p>
              <p>Feeling: {result?.feeling}</p>
              <p>Reliability:</p>
              {reliabilities} 
              </>} 
            {result == 0 && <p>Loading...</p>}
            {result == null && <p>Results will be here</p>}
          </div>
        </div>
        <div className='w-full flex flex-col items-center'>
          <h2 className='font-semibold text-2xl' style={isDark ? {color: "white"} : {}}>Architectures Comparison</h2>
          <div className='h-[50vh] w-5/6 flex justify-center items-center'>
            <ResponsiveLine
              data={data}
              margin={{ top: 50, right: 110, bottom: 50, left: 60 }}
              xScale={{type: "point"}}
              yScale={{
                type: 'linear',
                min: 'auto',
                max: 'auto',
                stacked: false,
                reverse: false
              }}
              curve="catmullRom"
              yFormat=" >-.2f"
              axisTop={null}
              axisRight={null}
              axisBottom={{
                  tickSize: 5,
                  tickPadding: 5,
                  tickRotation: 0,
                  legend: 'epoch',
                  legendOffset: 36,
                  legendPosition: 'middle'
              }}
              axisLeft={{
                tickSize: 5,
                tickPadding: 5,
                tickRotation: 0,
                legend: 'accuracy',
                legendOffset: -40,
                legendPosition: 'middle'
              }}
              pointSize={10}
              pointColor={{ theme: 'background' }}
              theme={isDark ? {
                "background": "transparent",
                "textColor": "#fff",
                "fontSize": 11,
                "axis": {
                    "domain": {
                        "line": {
                            "stroke": "#777777",
                            "strokeWidth": 1
                        }
                    },
                    "legend": {
                        "text": {
                            "fontSize": 12,
                            "fill": "#fff"
                        }
                    },
                    "ticks": {
                        "line": {
                            "stroke": "#777777",
                            "strokeWidth": 1
                        },
                        "text": {
                            "fontSize": 11,
                            "fill": "#fff"
                        }
                    }
                },
                "grid": {
                    "line": {
                        "stroke": "#dddddd",
                        "strokeWidth": 1
                    }
                },
                "legends": {
                    "title": {
                        "text": {
                            "fontSize": 11,
                            "fill": "#fff"
                        }
                    },
                    "text": {
                        "fontSize": 11,
                        "fill": "#fff"
                    },
                    "ticks": {
                        "line": {},
                        "text": {
                            "fontSize": 10,
                            "fill": "#333333"
                        }
                    }
                },
                "annotations": {
                    "text": {
                        "fontSize": 13,
                        "fill": "#333333",
                        "outlineWidth": 2,
                        "outlineColor": "#ffffff",
                        "outlineOpacity": 1
                    },
                    "link": {
                        "stroke": "#000000",
                        "strokeWidth": 1,
                        "outlineWidth": 2,
                        "outlineColor": "#ffffff",
                        "outlineOpacity": 1
                    },
                    "outline": {
                        "stroke": "#000000",
                        "strokeWidth": 2,
                        "outlineWidth": 2,
                        "outlineColor": "#ffffff",
                        "outlineOpacity": 1
                    },
                    "symbol": {
                        "fill": "#000000",
                        "outlineWidth": 2,
                        "outlineColor": "#ffffff",
                        "outlineOpacity": 1
                    }
                },
                "tooltip": {
                    "container": {
                        "background": "#ffffff",
                        "color": "#333333",
                        "fontSize": 12
                    },
                    "basic": {},
                    "chip": {},
                    "table": {},
                    "tableCell": {},
                    "tableCellValue": {}
                }
              } : {
                "background": "#ffffff",
                "textColor": "#333333",
                "fontSize": 11,
                "axis": {
                    "domain": {
                        "line": {
                            "stroke": "#777777",
                            "strokeWidth": 1
                        }
                    },
                    "legend": {
                        "text": {
                            "fontSize": 12,
                            "fill": "#333333"
                        }
                    },
                    "ticks": {
                        "line": {
                            "stroke": "#777777",
                            "strokeWidth": 1
                        },
                        "text": {
                            "fontSize": 11,
                            "fill": "#333333"
                        }
                    }
                },
                "grid": {
                    "line": {
                        "stroke": "#dddddd",
                        "strokeWidth": 1
                    }
                },
                "legends": {
                    "title": {
                        "text": {
                            "fontSize": 11,
                            "fill": "#333333"
                        }
                    },
                    "text": {
                        "fontSize": 11,
                        "fill": "#333333"
                    },
                    "ticks": {
                        "line": {},
                        "text": {
                            "fontSize": 10,
                            "fill": "#333333"
                        }
                    }
                },
                "annotations": {
                    "text": {
                        "fontSize": 13,
                        "fill": "#333333",
                        "outlineWidth": 2,
                        "outlineColor": "#ffffff",
                        "outlineOpacity": 1
                    },
                    "link": {
                        "stroke": "#000000",
                        "strokeWidth": 1,
                        "outlineWidth": 2,
                        "outlineColor": "#ffffff",
                        "outlineOpacity": 1
                    },
                    "outline": {
                        "stroke": "#000000",
                        "strokeWidth": 2,
                        "outlineWidth": 2,
                        "outlineColor": "#ffffff",
                        "outlineOpacity": 1
                    },
                    "symbol": {
                        "fill": "#000000",
                        "outlineWidth": 2,
                        "outlineColor": "#ffffff",
                        "outlineOpacity": 1
                    }
                },
                "tooltip": {
                    "container": {
                        "background": "#ffffff",
                        "color": "#333333",
                        "fontSize": 12
                    },
                    "basic": {},
                    "chip": {},
                    "table": {},
                    "tableCell": {},
                    "tableCellValue": {}
                }
              }}
              pointBorderWidth={2}
              pointBorderColor={{ from: 'serieColor' }}
              pointLabelYOffset={-12}
              useMesh={true}
              legends={[
                {
                    anchor: 'bottom-right',
                    direction: 'column',
                    justify: false,
                    translateX: 100,
                    translateY: 0,
                    itemsSpacing: 0,
                    itemDirection: 'left-to-right',
                    itemWidth: 80,
                    itemHeight: 20,
                    itemOpacity: 0.75,
                    symbolSize: 12,
                    symbolShape: 'circle',
                    symbolBorderColor: 'rgba(0, 0, 0, .5)',
                    effects: [
                        {
                            on: 'hover',
                            style: {
                                itemBackground: 'rgba(0, 0, 0, .03)',
                                itemOpacity: 1
                            }
                        }
                    ]
                }
              ]}
            />  
          </div>
        </div>
      
      </div>
    </>
  )
}

export default Playground;