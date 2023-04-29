"use client"

import React, { useState } from 'react';
import { ResponsiveLine, Serie } from '@nivo/line';
import Papa from 'papaparse';
import { useSearchParams } from 'next/navigation';

const Train = () => {
  const isDark = localStorage.getItem("mode") == "Dark";

  const [text, setText] = useState("");
  const [result, setResult] = useState<any>(null);
  const [reliabilities, setReliabilities] = useState<any>();

  const [currentArch, setCurrentArch] = useState<string>("");

  const architectureChosen = (e:any) => {
    setCurrentArch(e.target.innerText);
  }

  const parseData = () =>{
    let array : Serie[] = [];

    Papa.parse("assets/GRU-accuracy.csv", {
        download: true,
        complete: function(results) {
          const data : any = results.data;
          
          let final = []
          for(let i = 1; i < data.length - 1; i++) {
            final.push({x: i, y: data[i][1]});
          }
          const nivoData = {
            id: "GRU",
            color: "hsl(54, 70%, 50%)",
            data: final
          }
          array.push(nivoData);
        }
    });
    Papa.parse("assets/LSTM-accuracy.csv", {
      download: true,
      complete: function(results) {
        const data : any = results.data;
        
        let final = []
        for(let i = 1; i < data.length - 1; i++) {
          final.push({x: i, y: data[i][1]});
        }
        const nivoData = {
          id: "LSTM",
          color: "hsl(141, 70%, 50%)",
          data: final
        }
        array.push(nivoData);
      }
    });
    console.log(array);
    return array;
  }

  const [data, setData] = useState(parseData());

  const click = async () => {
    setResult(0);
    setReliabilities(null);
    const res = await fetch("http://localhost:8080/sentiment?arch=" + currentArch + "&sentence=" + text);
    const json = await res.json();
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
  <div className='h-full w-full flex flex-row-reverse items-center justify-center gap-2 font-[Poppins] z-50' style={isDark ? { backgroundColor: 'black' } : { backgroundColor: 'white' }}>
    {/* <p>{text}</p> */}
    <div className='flex flex-col justify-center items-center gap-2 h-1/2 w-fit px-52 border-l-solid border-l-white border-l-[1px]' style={isDark ? {borderLeft: "1px solid white"} : {borderLeft: "1px solid black"}}>
      <h1 className='text-2xl font-bold' style={isDark ? {color: "white"} : {}}>Playground</h1>
      <input className='rounded-md p-2' style={isDark ? {color: "white", backgroundColor: "black", border: "1px solid white"} : {border: "1px solid black"}} placeholder='Text' onChange={(e) => setText(e.target.value)}/>

      <div className='flex gap-2'>
        <button onClick={architectureChosen} className="border-solid border-[1px] border-black rounded-md p-2" style={currentArch == "GRU" ? {backgroundColor: "rgba(211,211,211,1)"} : {backgroundColor: "white"}}>GRU</button>
        <button onClick={architectureChosen} className="border-solid border-[1px] border-black rounded-md p-2" style={currentArch == "LSTM" ? {backgroundColor: "rgba(211,211,211,1)"} : {backgroundColor: "white"}}>LSTM</button>
      </div>

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
  )
}

export default Train;